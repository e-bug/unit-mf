#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created by:         Emanuele Bugliarello (@e-bug)
Date created:       5/24/2019
Date last modified: 5/24/2019
"""

from __future__ import absolute_import, division, print_function

import os 
import pickle
import numpy as np
import pandas as pd


# ============================================================================ #
#                                  Load data                                   #
# ============================================================================ #

print('Loading data...')

# Check data has been downloaded
data_dir = '../data'
page_views_sample_fn = os.path.join(data_dir, 'page_views_sample.gz')
clicks_train_fn = os.path.join(data_dir, 'clicks_train.gz')
events_fn = os.path.join(data_dir, 'events.gz')
docs_cats_fn = os.path.join(data_dir, 'documents_categories.gz')

exists = 1
for fn in [page_views_sample_fn, clicks_train_fn, events_fn, docs_cats_fn]:
    exists *= os.path.isfile(fn)
if exists == 0:
    print("""From https://www.kaggle.com/c/outbrain-click-prediction/data, 
             download:
             - page_views_sample.csv
             - clicks_train.csv
             - events.csv
             - documents_categories.csv
             Then compress them using gzip into %s as:
             - %s
             - %s
             - %s
             - %s
          """ % (data_dir, page_views_sample_fn, clicks_train_fn, 
                 events_fn, docs_cats_fn))
    exit(1)

# Load page views sample
page_views = pd.read_csv(page_views_sample_fn, header=0, 
                         usecols=[0,1], compression='gzip')
print('\tLoaded page_views. Size:', page_views.shape)

# Load clicks
clicks = pd.read_csv(clicks_train_fn, header=0, compression='gzip')
print('\tLoaded clicks. Size:', clicks.shape)
print('\t\tNumber of unique ads: %d' % clicks.ad_id.nunique())
print('\t\tNumber of unique displays: %d' % clicks.display_id.nunique())

# Load events
events = pd.read_csv(events_fn, header=0, compression='gzip')
print('\tLoaded events. Size:', events.shape)
print('\t\tNumber of unique displays: %d' % events.display_id.nunique())
print('\t\tNumber of unique documents: %d' % events.document_id.nunique())

# Load documents categories
docs_cats = pd.read_csv(docs_cats_fn, header=0, 
                        compression='gzip')
print('\tLoaded document_categories. Size:', docs_cats.shape)
print('\t\tNumber of unique categories: %d' % 
      docs_cats[docs_cats.confidence_level >= 0.9].category_id.nunique())


# ============================================================================ #
#                                Aggregate data                                #
# ============================================================================ #

print('Aggregating data. This may take a while...')

# Add which document (document_id) each display (display_id) is in
merge = clicks.merge(events.drop(['uuid', 'timestamp', 
                                  'platform', 'geo_location'], axis=1))

# Add the category of each document if its confidence level is at least 0.9
merge = merge.merge(docs_cats[docs_cats.confidence_level >= 0.9]
                                       .drop('confidence_level', axis=1))

# Drop unnecessary columns and order them
merge.drop(['display_id', 'document_id'], axis=1, inplace=True)
merge = merge[['ad_id', 'category_id', 'clicked']]

# Map IDs to indices
adId2idx = {u: i for i,u in enumerate(sorted(merge.ad_id.drop_duplicates()))}
catId2idx = {u: i for i,u in enumerate(sorted(
                                       merge.category_id.drop_duplicates()))}
merge['ad_id'] = merge.ad_id.apply(lambda u: adId2idx[u])
merge['category_id'] = merge.category_id.apply(lambda m: catId2idx[m])

# Average clicks for ad in document category
avgdf = merge.groupby(['ad_id', 'category_id']).mean().reset_index()

# Sum clicks for ad in document category
sumdf = merge.groupby(['ad_id', 'category_id']).count().reset_index()

# Data sizes
n_ads = len(avgdf.ad_id.drop_duplicates())
n_cats = len(avgdf.category_id.drop_duplicates())
n_entries = avgdf.shape[0]

# Create matrix of clicks sums
m = -1 * np.ones((n_ads, n_cats))
for _, r in sumdf.iterrows():
    m[int(r['ad_id']), int(r['category_id'])] = r['clicked']
sum_df = pd.DataFrame(m)

# Create matrix of clicks averages
m = -1 * np.ones((n_ads, n_cats))
for _,r in avgdf.iterrows():
    m[int(r['ad_id']), int(r['category_id'])] = r['clicked']
avg_df = pd.DataFrame(m)


# ============================================================================ #
#                                  Clean data                                  #
# ============================================================================ #

print('Cleaning data...')

# Remove entries displayed less than 10 times in a document category
min_displays = 9
mask = sum_df > min_displays
sum_df = sum_df[mask]
avg_df = avg_df[mask]

# Remove categories with < 10 entries - Remove ads with < 5 entries
min_cats = 9
min_ads = 4
eff_df = avg_df
nclaims_df = sum_df
previous_shape = None
current_shape = eff_df.shape
while previous_shape != current_shape:
    # Remove disciplines with less than 10 entries
    filt_cols = eff_df.count() > min_cats
    rm_cols = []
    for col in eff_df.columns:
        if filt_cols[col] == False:
            rm_cols.append(col)
    eff_df.drop(rm_cols, axis=1, inplace=True)
    nclaims_df.drop(rm_cols, axis=1, inplace=True)
    # Remove users with less than 5 entries
    n_entries_peruser = eff_df.count(axis=1)
    eff_df = eff_df[n_entries_peruser > min_ads]
    nclaims_df = nclaims_df[n_entries_peruser > min_ads]
    # Update shapes
    previous_shape = current_shape
    current_shape = [eff_df.shape[0], eff_df.shape[1]]
# Reset indices
eff_df.reset_index(inplace=True)
eff_df.drop('index', axis=1, inplace=True)
nclaims_df.reset_index(inplace=True)
nclaims_df.drop('index', axis=1, inplace=True)


# ============================================================================ #
#                                  Store data                                  #
# ============================================================================ #

print('Saving final data...')

# Save filtered DataFrames as gzip'ed CSV files
eff_df.to_csv(os.path.join(data_dir, 'efficiency.gz'), sep=',', header=False, 
              index=False, na_rep=-1, compression='gzip')
nclaims_df.to_csv(os.path.join(data_dir, 'nimpressions.gz'), sep=',', 
                  header=False, index=False, na_rep=-1, compression='gzip')

print('Finished!')
assert eff_df.shape == (15647, 85), "Note: Data size does not match with ours"


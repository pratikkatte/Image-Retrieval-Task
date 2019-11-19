import os
import sys
import argparse
import threading

import numpy as np

from src.features import (compute_features, compute_representation,
                          compute_localization_representation)
from src.search.search_model import SearchModel
from src.search.localization import localize

def _descending_argsort(array, k):
    indices = np.argpartition(array, -k)[-k:]
    indices = indices[np.argsort(array[indices])]  # Sort indices
    return indices[::-1]


def _query(query_features, feature_store, top_n=0):

    # Cosine similarity measure
    similarity = feature_store.dot(query_features.T).flatten()

    k = min(len(feature_store), abs(top_n))
    if top_n >= 0:  # Best top_n features
        indices = _descending_argsort(similarity, k)


    return indices, similarity[indices]


def _localize_parallel(search_model, query_features, feature_idxs, image_shape,
                       n_threads=8):

    def f(res, feature_idxs):
        for idx, feature_idx in enumerate(feature_idxs):
            features = search_model.get_features(feature_idx)
            res[idx] = localize(localization_repr, features, image_shape)

    localization_repr = compute_localization_representation(query_features)
    bounding_boxes = np.empty(len(feature_idxs), dtype=(int, 4))

    threads = []
    chunk_len = int(np.ceil(len(feature_idxs) / n_threads))
    for i in range(n_threads):
        args = [bounding_boxes[i * chunk_len:(i + 1) * chunk_len],
                feature_idxs[i * chunk_len:(i + 1) * chunk_len]]
        threads.append(threading.Thread(target=f, args=args))

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    return bounding_boxes


def _localize(search_model, query_features, feature_idxs, image_shape):

    localization_repr = compute_localization_representation(query_features)
    bounding_boxes = np.empty(len(feature_idxs), dtype=(int, 4))
    for idx, feature_idx in enumerate(feature_idxs):
        features = search_model.get_features(feature_idx)
        bounding_boxes[idx] = localize(localization_repr, features, image_shape)

    return bounding_boxes


def _compute_bbox_reprs(search_model, bounding_boxes, feature_idxs):

    repr_size = search_model.feature_store.shape[-1]
    bounding_box_reprs = np.empty((len(feature_idxs), repr_size))
    for idx, bbox in enumerate(bounding_boxes):
        x1, y1, x2, y2 = bbox
        features = search_model.get_features(feature_idxs[idx])
        bbox_repr = compute_representation(features[y1:y2+1, x1:x2+1],
                                           search_model.pca)
        bounding_box_reprs[idx] = bbox_repr
    return bounding_box_reprs


def _average_query_exp(query_repr, feature_reprs, feature_idxs, top_n=5):
    reprs = np.vstack((feature_reprs[feature_idxs[:top_n]], query_repr))
    avg_repr = np.average(reprs, axis=0)
    indices, similarities = _query(avg_repr, feature_reprs)
    return indices, similarities


def _map_bboxes(search_model, bboxes, features_idxs):

    mapped_bboxes = []
    for bbox, feature_idx in zip(bboxes, features_idxs):
        metadata = search_model.get_metadata(feature_idx)
        scale_x = metadata['width'] / float(metadata['feature_width'])
        scale_y = metadata['height'] / float(metadata['feature_height'])
        mapped_bboxes.append((round(bbox.item(0)*scale_x),
                              round(bbox.item(1)*scale_y),
                              round(bbox.item(2)*scale_x),
                              round(bbox.item(3)*scale_y)))
    return mapped_bboxes


def search(search_model, query, top_n=0, localize=True, localize_n=50,
           rerank=True, avg_qe=True):

    assert top_n >= 0
    if rerank:
        assert localize, 'Rerank implies localization'

    bboxes = None
    reprs = search_model.feature_store

    query_features = compute_features(search_model.model, query)
    query_repr = compute_representation(query_features, search_model.pca)

    retrieval_n = localize_n if localize else 0
    feature_idxs, sims = _query(query_repr, reprs, retrieval_n)
    idxs = feature_idxs

    if localize:
        bboxes = _localize_parallel(search_model, query_features, idxs,
                                    query.shape[:2])

    if rerank:
        reprs = _compute_bbox_reprs(search_model, bboxes, idxs)
        idxs, sims = _query(query_repr, reprs)

    if avg_qe:
        idxs, sims = _average_query_exp(query_repr, reprs, idxs)

    if top_n > 0:
        idxs = idxs[:top_n]

    if localize:
        bboxes = _map_bboxes(search_model, bboxes[idxs], feature_idxs[idxs])

    return feature_idxs[idxs], sims, bboxes

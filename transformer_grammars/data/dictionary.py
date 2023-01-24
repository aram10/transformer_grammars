# Copyright 2021-2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Defines a dictionary data structure.

The dictionary data structure is a bidirectional one that converts:
  i. Word types to embedding indices, and
  ii. Embedding indices to word types.
The dictionary can be evoked like a native Python dictionary, e.g.
dict['cat'] = 5 and dict[5] = 'cat'.
If a word type does not exist in the dictionary, a new index will be created.
Functions like UNK-ification and freezing are handled
separately by the Unkifier class.
"""

import collections

from absl import logging
import numpy as np


class Dict(object):
  """Dictionary class to convert word types to embedding indices."""

  def __init__(self):
    """Initialize the dictionary object.

    Args:

    Returns:
      None.
    """
    # Map string to indices.
    self.map = collections.defaultdict(lambda: len(self.map))

    # Map indices to string using a list of words in the dictionary.
    self.map_rev = []

    # Boolean to to indicate if dictionary is frozen.
    self.frozen = False

  def __len__(self):
    """Obtain the size (number of words) in the current dictionary.

    Args:

    Returns:
      An integer >= 0.
    """
    return len(self.map)

  def __contains__(self, word):
    """Check whether the dictionary contains a particular word.

    Args:
      word: A string that may or may not exist in the dictionary.

    Returns:
      A boolean that specifies whether the word exists in the dictionary.
    """
    return word in self.map

  def freeze(self):
    """Freeze the dictionary to prevent conversion of new word types.

    Args:

    Returns:
      None.
    """
    self.frozen = True

  def __getitem__(self, item):
    """Convert a word to its index, or an index to its string word form.

    Args:
      item: either a string word type or an integer embedding index.

    Returns:
      either the corresponding embedding index or the string form of the
      item.

    Raises:
      IndexError: converting an out-of-bounds embedding index.
      ValueError: wrong argument type or converting a new type when frozen.
    """
    if isinstance(item, str):
      if self.frozen and item not in self.map:
        raise ValueError(f"Converting a new type: {item} when frozen.")

      # Retrieve item's embedding index (if existent) or create a new one.
      emb_idx = self.map[item]

      # Populate the reverse dictionary if necessary.
      if emb_idx >= len(self.map_rev):
        self.map_rev.append(item)
        # Assert that the we can retrieve the correct word given the index.
        assert self.map_rev[emb_idx] == item
      return emb_idx
    elif isinstance(item, (int, np.integer)):
      # item is an int, retrieve its string word form.
      if item < 0:
        raise IndexError("Indices in the dictionary are >= 0")
      return self.map_rev[item]
    else:
      raise ValueError("The passed argument is neither string nor integer.")

  def clear(self):
    """Clear the internal dictionary elements.

    Args:

    Returns:
      None.
    """
    self.map.clear()
    self.map = collections.defaultdict(lambda: len(self.map))

  def items(self):
    """Get the iterator over the (key, value) pairs in the Dict object.

    Args:

    Returns:
      An iterator over (key, value) pairs.
    """
    return self.map.items()

  def values(self):
    """Get the iterator over the (non-unique) values in the Dict object.

    Args:

    Returns:
      An iterator over each value entry in the Dict object.
    """
    return self.map.values()

  def load_from_file(self, file_obj):
    """Load vocabulary from file.

    Args:
      file_obj: A file object that represents the vocabulary file.

    Returns:
      None (the dictionary is populated).
    """
    lines_ctr = 0
    for line in file_obj:
      lines_ctr += 1
      word = line.rstrip()
      _ = self[word]
    logging.info("Read %s lines from the file", lines_ctr)

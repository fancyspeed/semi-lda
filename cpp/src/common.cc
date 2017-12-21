// Copyright 2008 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "common.h"

char kSegmentFaultCauser[] = "Used to cause artificial segmentation fault";

namespace learning_lda {

bool IsValidProbDistribution(const TopicProbDistribution& dist) {
  const double kUnificationError = 0.00001;
  double sum_distribution = 0;
  for (int k = 0; k < dist.size(); ++k) {
    sum_distribution += dist[k];
  }
  return (sum_distribution - 1) * (sum_distribution - 1)
      <= kUnificationError;
}

int LoadWordIndex(const string& word_index_file,
                  map<string, int>& word_index_map) {
  word_index_map.clear();
  ifstream fin(word_index_file.c_str());
  string line;
  int32 id;
  string word;
  while (getline(fin, line)) {
    if (line.size() > 0 &&     // Skip empty lines.
        line[0] != '\r' &&     // Skip empty lines.
        line[0] != '\n' &&     // Skip empty lines.
        line[0] != '#') {      // Skip empty lines.
      istringstream ss(line);
      if (ss >> id >> word) {
        word_index_map.insert(make_pair(word, id));
      } 
    }
  } 
  return word_index_map.size();
}

int LoadSeedWords(const string& seed_word_file,
                  int num_topics,
                  const map<string, int>& word_index_map,
                  map<int, vector<int32> >& word_topics_map) {
  word_topics_map.clear();
  vector<int32> default_topics;
  for (int i=0; i<num_topics; ++i) {
    default_topics.push_back(i);
  }
  word_topics_map.insert(make_pair(-1, default_topics));

  if (seed_word_file.empty()) {
    return word_topics_map.size();
  }
  ifstream fin(seed_word_file.c_str());
  string line;
  while (getline(fin, line)) { // Each line is for a topic.
    if (line.size() > 0 &&     // Skip empty lines.
        line[0] != '\r' &&     // Skip empty lines.
        line[0] != '\n' &&     // Skip empty lines.
        line[0] != '#') {      // Skip empty lines.
      istringstream ss(line);
      int32 topic;
      string cid;
      ss >> topic >> cid;
      if (topic >= num_topics) {
        continue;
      }
      string word;
      while (ss >> word) {
        if (word_index_map.end() == word_index_map.find(word)) {
          continue;
        }
        int word_index = word_index_map.find(word)->second;
        if (word_topics_map.end() == word_topics_map.find(word_index)) {
          vector<int32> empty_list;
          word_topics_map.insert(make_pair(word_index, empty_list));
        }
        word_topics_map.find(word_index)->second.push_back(topic);
      }
    }
  }
  return word_topics_map.size();
}

int GetAccumulativeSample(const vector<double>& distribution) {
  double distribution_sum = 0.0;
  for (int i = 0; i < distribution.size(); ++i) {
    distribution_sum += distribution[i];
  }

  double choice = RandDouble() * distribution_sum;
  double sum_so_far = 0.0;
  for (int i = 0; i < distribution.size(); ++i) {
    sum_so_far += distribution[i];
    if (sum_so_far >= choice) {
      return i;
    }
  }

  LOG(FATAL) << "Failed to choose element from distribution of size "
             << distribution.size() << " and sum " << distribution_sum;

  return -1;
}

std::ostream& operator << (std::ostream& out, vector<double>& v) {
  for (size_t i = 0; i < v.size(); ++i) {
    out << v[i] << " ";
  }
  return out;
}

}  // namespace learning_lda

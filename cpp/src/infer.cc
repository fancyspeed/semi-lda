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
/*
  An example running of this program:

  ./infer \
  --alpha 0.1    \
  --beta 0.01                                           \
  --inference_data_file ./testdata/test_data.txt \
  --inference_result_file /tmp/inference_result.txt \
  --model_file /tmp/lda_model.txt                       \
  --burn_in_iterations 10                              \
  --total_iterations 15
*/
#include <fstream>
#include <set>
#include <sstream>
#include <string>

#include "common.h"
#include "document.h"
#include "model.h"
#include "sampler.h"
#include "cmd_flags.h"

int main(int argc, char** argv) {
  using learning_lda::LoadSeedWords;
  using learning_lda::LDACorpus;
  using learning_lda::LDAModel;
  using learning_lda::LDAAccumulativeModel;
  using learning_lda::LDASampler;
  using learning_lda::LDADocument;
  using learning_lda::LDACmdLineFlags;
  using learning_lda::DocumentWordTopicsPB;
  using learning_lda::RandInt;
  using std::ifstream;
  using std::ofstream;
  using std::istringstream;

  LDACmdLineFlags flags;
  flags.ParseCmdFlags(argc, argv);
  if (!flags.CheckInferringValidity()) {
    return -1;
  }

  map<string, int> word_index_map;
  ifstream model_fin(flags.model_file_.c_str());
  LDAModel model(model_fin, &word_index_map);
  map<int, string> index_word_map;
  for (map<string, int>::iterator iter = word_index_map.begin();
       iter != word_index_map.end();
       ++iter) {
    index_word_map[iter->second] = iter->first;
  }

  map<int, vector<int32> > word_topics_map;
  LoadSeedWords(flags.seed_word_file_, 
                model.num_topics(),
                word_index_map,
                word_topics_map);

  //srand(time(NULL));
  srand(1000001);
  LDASampler sampler(flags.alpha_, flags.beta_, &model, NULL, &word_topics_map);

  ifstream fin(flags.inference_data_file_.c_str());
  ofstream out(flags.inference_result_file_.c_str());
  if (flags.word_assignments_file_.empty()) {
    flags.word_assignments_file_ = flags.inference_result_file_ + ".assignment";
  }
  ofstream out2(flags.word_assignments_file_.c_str());

  string line;
  while (getline(fin, line)) {  // Each line is a training document.
    if (line.size() > 0 &&      // Skip empty lines.
        line[0] != '\r' &&      // Skip empty lines.
        line[0] != '\n' &&      // Skip empty lines.
        line[0] != '#') {       // Skip comment lines.
      DocumentWordTopicsPB document_topics;
      istringstream ss(line);
      string word;
      int count;
      while (ss >> word >> count) {  // Load and init a document.
        vector<int32> topics;
        for (int i = 0; i < count; ++i) {
          topics.push_back(RandInt(model.num_topics()));
        }
        map<string, int>::const_iterator iter = word_index_map.find(word);
        if (iter != word_index_map.end()) {
          document_topics.add_wordtopics(word, iter->second, topics);
        }
      }
      vector<int32> label_topics;
      LDADocument document(document_topics, model.num_topics(), label_topics);

      TopicProbDistribution prob_dist(model.num_topics(), 0);
      for (int iter = 0; iter < flags.total_iterations_; ++iter) {
        sampler.SampleNewTopicsForDocument(&document, false);
        if (iter >= flags.burn_in_iterations_) {
          const vector<int64>& document_distribution =
              document.topic_distribution();
          for (int i = 0; i < document_distribution.size(); ++i) {
            prob_dist[i] += document_distribution[i];
          }
        }
      }

      for (int topic = 0; topic < prob_dist.size(); ++topic) {
        out << prob_dist[topic] /
              (flags.total_iterations_ - flags.burn_in_iterations_)
            << ((topic < prob_dist.size() - 1) ? " " : "");
      }
      map<int, int> word_assign;
      document.word_assignments(word_assign);
      for (map<int, int>::iterator iter = word_assign.begin();
           iter != word_assign.end();
           ++iter) {
        out2 << index_word_map[iter->first] << ":" << iter->second << " ";
      }
      out << "\n";
      out2 << "\n";
    } else {
      out << "\n";
      out2 << "\n";
    }
  }  // while (getline(fin, line))
}  // main


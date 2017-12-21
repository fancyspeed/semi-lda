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

  ./lda           \
  --num_topics 2 \
  --alpha 0.1    \
  --beta 0.01                                           \
  --training_data_file ./testdata/test_data.txt \
  --model_file /tmp/lda_model.txt                       \
  --burn_in_iterations 100                              \
  --total_iterations 150
*/

#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <map>

#include "common.h"
#include "document.h"
#include "model.h"
#include "accumulative_model.h"
#include "sampler.h"
#include "cmd_flags.h"

namespace learning_lda {

using std::ifstream;
using std::ofstream;
using std::istringstream;
using std::vector;
using std::set;
using std::map;

int LoadAndInitTrainingCorpus(
    const string& corpus_file,
    int num_topics,
    const map<string, int>& word_index_map,
    const map<int, vector<int32> >& word_topics_map,
    LDACorpus* corpus) {
  corpus->clear();
  ifstream fin(corpus_file.c_str());
  string line;
  while (getline(fin, line)) {  // Each line is a training document.
    if (line.size() > 0 &&      // Skip empty lines.
        line[0] != '\r' &&      // Skip empty lines.
        line[0] != '\n' &&      // Skip empty lines.
        line[0] != '#') {       // Skip comment lines.
      string word;
      int count;

      vector<int32> candidate_topics;
      for (int i=0; i<num_topics; ++i) {
        candidate_topics.push_back(i);
      }
      istringstream ss(line);
      while (ss >> word >> count){  // Modify candidate topics
        if (word_index_map.end() == word_index_map.find(word))
          continue;
        int word_index = word_index_map.find(word)->second;
        if (word_topics_map.end() != word_topics_map.find(word_index)) {
          vector<int32> seed_topics = word_topics_map.find(word_index)->second;
          for (int i = 0; i < count; ++i) {
            candidate_topics.push_back(
                seed_topics[RandInt(seed_topics.size())]);
          }
        }
      }
      istringstream ss2(line);
      DocumentWordTopicsPB document;
      while (ss2 >> word >> count) {  // Load and init a document.
        if (word_index_map.end() == word_index_map.find(word))
          continue;
        int word_index = word_index_map.find(word)->second;

        vector<int32> topics;
        if (word_topics_map.end() != word_topics_map.find(word_index)) {
          vector<int32> seed_topics = word_topics_map.find(word_index)->second;
          for (int i = 0; i < count; ++i) {
            topics.push_back(
                seed_topics[RandInt(seed_topics.size())]);
          }
        } else {
          for (int i = 0; i < count; ++i) {
            topics.push_back(
                candidate_topics[RandInt(candidate_topics.size())]);
          }
        }
        document.add_wordtopics(word, word_index, topics);
      }
      corpus->push_back(new LDADocument(document, num_topics));
    }
  }
  return corpus->size();
}

void FreeCorpus(LDACorpus* corpus) {
  for (list<LDADocument*>::iterator iter = corpus->begin();
       iter != corpus->end();
       ++iter) {
    if (*iter != NULL) {
      delete *iter;
      *iter = NULL;
    }
  }
}

}  // namespace learning_lda

int main(int argc, char** argv) {
  using learning_lda::LDACorpus;
  using learning_lda::LDAModel;
  using learning_lda::LDAAccumulativeModel;
  using learning_lda::LDASampler;
  using learning_lda::LDADocument;
  using learning_lda::LoadSeedWords;
  using learning_lda::LoadWordIndex;
  using learning_lda::LoadAndInitTrainingCorpus;
  using learning_lda::LDACmdLineFlags;
  using std::list;
  using std::vector;
  using std::set;

  LDACmdLineFlags flags;
  flags.ParseCmdFlags(argc, argv);
  if (!flags.CheckTrainingValidity()) {
    return -1;
  }

  map<string, int> word_index_map;
  if (!flags.init_model_file_.empty()) {
    ifstream model_fin(flags.init_model_file_.c_str());
    LDAModel old_model(model_fin, &word_index_map);
    std::cout << "after init model, word_index_map size: " << int(word_index_map.size()) << std::endl;
  } else if (!flags.word_index_file_.empty()) {
    int n_words = LoadWordIndex(flags.word_index_file_, word_index_map);
    std::cout<<"word_index_map size: "<<n_words<<std::endl;
  } else {
    std::cerr << "both init_model_file and word_index_map are empty!" << std::endl;
    return -1;
  }

  map<int, vector<int32> > word_topics_map;
  LoadSeedWords(flags.seed_word_file_, 
                flags.num_topics_,
                word_index_map,
                word_topics_map);

  srand(time(NULL));
  LDACorpus corpus;
  CHECK_GT(LoadAndInitTrainingCorpus(flags.training_data_file_,
                                     flags.num_topics_,
                                     word_index_map,
                                     word_topics_map,
                                     &corpus), 0);
  std::cout << "Training data loaded" << std::endl;

  if (!flags.init_model_file_.empty()) {
    ifstream model_fin(flags.init_model_file_.c_str());
    LDAModel old_model(model_fin, &word_index_map);
    LDASampler old_sampler(flags.alpha_, flags.beta_, &old_model, NULL, &word_topics_map);
    for (int iter=0; iter<flags.burn_in_iterations_; ++iter) {
      std::cout << "Iteration using old model " << iter << " ...\n";
      old_sampler.DoIteration(&corpus, false, iter<flags.burn_in_iterations_);
    }
  }

  LDAModel model(flags.num_topics_, word_index_map);
  LDAAccumulativeModel accum_model(flags.num_topics_, word_index_map.size());
  LDASampler sampler(flags.alpha_, flags.beta_, &model, &accum_model, &word_topics_map);
  sampler.InitModelGivenTopics(corpus);

  for (int iter = 0; iter < flags.total_iterations_; ++iter) {
    std::cout << "Iteration " << iter << " ...\n";
    if (flags.compute_likelihood_ == "true") {
      double loglikelihood = 0;
      for (list<LDADocument*>::const_iterator iterator = corpus.begin();
           iterator != corpus.end();
           ++iterator) {
        loglikelihood += sampler.LogLikelihood(*iterator);
      }
      std::cout << "Loglikelihood: " << loglikelihood << std::endl;
    }

    sampler.DoIteration(&corpus, true, iter < flags.burn_in_iterations_);
  }

  accum_model.AverageModel(
      flags.total_iterations_ - flags.burn_in_iterations_);

  FreeCorpus(&corpus);

  std::ofstream fout(flags.model_file_.c_str());
  accum_model.AppendAsString(word_index_map, fout);

  return 0;
}


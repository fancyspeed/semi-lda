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

  mpiexec -n 2 ./mpi_lda           \
  --num_topics 2 \
  --alpha 0.1    \
  --beta 0.01                                           \
  --training_data_file ./testdata/test_data.txt \
  --model_file /tmp/lda_model.txt                       \
  --burn_in_iterations 100                              \
  --total_iterations 150
*/

#include "mpi.h"

#include <algorithm>
#include <fstream>
#include <set>
#include <vector>
#include <sstream>
#include <string>
#include <cstdlib>

#include "common.h"
#include "document.h"
#include "model.h"
#include "accumulative_model.h"
#include "sampler.h"
#include "cmd_flags.h"

using std::ifstream;
using std::ofstream;
using std::istringstream;
using std::set;
using std::vector;
using std::list;
using std::map;
using std::sort;
using std::string;
using learning_lda::LDADocument;

namespace learning_lda {

// A wrapper of MPI_Allreduce. If the vector is over 32M, we allreduce part
// after part. This will save temporary memory needed.
void AllReduceTopicDistribution(int64* buf, int count) {
  static int kMaxDataCount = 1 << 22;
  static int datatype_size = sizeof(*buf);
  if (count > kMaxDataCount) {
    char* tmp_buf = new char[datatype_size * kMaxDataCount];
    for (int i = 0; i < count / kMaxDataCount; ++i) {
      MPI_Allreduce(reinterpret_cast<char*>(buf) +
             datatype_size * kMaxDataCount * i,
             tmp_buf,
             kMaxDataCount, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
      memcpy(reinterpret_cast<char*>(buf) +
             datatype_size * kMaxDataCount * i, 
             tmp_buf,
             kMaxDataCount * datatype_size);
    }
    // If count is not divisible by kMaxDataCount, there are some elements left
    // to be reduced.
    if (count % kMaxDataCount > 0) {
      MPI_Allreduce(reinterpret_cast<char*>(buf)
               + datatype_size * kMaxDataCount * (count / kMaxDataCount),
               tmp_buf,
               count - kMaxDataCount * (count / kMaxDataCount), MPI_LONG_LONG, MPI_SUM,
               MPI_COMM_WORLD);
      memcpy(reinterpret_cast<char*>(buf)
               + datatype_size * kMaxDataCount * (count / kMaxDataCount),
               tmp_buf,
               (count - kMaxDataCount * (count / kMaxDataCount)) * datatype_size);
    }
    delete[] tmp_buf;
  } else {
    char* tmp_buf = new char[datatype_size * count];
    MPI_Allreduce(buf, tmp_buf, count, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    memcpy(buf, tmp_buf, datatype_size * count);
    delete[] tmp_buf;
  }
}

class ParallelLDAModel : public LDAModel {
 public:
  ParallelLDAModel(int num_topic, const map<string, int>& word_index_map)
      : LDAModel(num_topic, word_index_map) {
  }
  void ComputeAndAllReduce(const LDACorpus& corpus) {
    for (list<LDADocument*>::const_iterator iter = corpus.begin();
         iter != corpus.end();
         ++iter) {
      LDADocument* document = *iter;
      for (LDADocument::WordOccurrenceIterator iter2(document);
           !iter2.Done(); iter2.Next()) {
        IncrementTopic(iter2.Word(), iter2.Topic(), 1);
      }
    }
    AllReduceTopicDistribution(&memory_alloc_[0], memory_alloc_.size());
  }
};

int DistributelyLoadAndInitTrainingCorpus(
    const string& corpus_file,
    int num_topics,
    int myid, int pnum, 
    const map<string, int>& word_index_map,
    const map<int, vector<int32> >& word_topics_map,
    LDACorpus* corpus) {
  corpus->clear();
  ifstream fin(corpus_file.c_str());
  string line;
  int index = 0;
  while (getline(fin, line)) {  // Each line is a training document.
    if (line.size() > 0 &&      // Skip empty lines.
        line[0] != '\r' &&      // Skip empty lines.
        line[0] != '\n' &&      // Skip empty lines.
        line[0] != '#') {       // Skip comment lines.

      if (index % pnum == myid) {
        // This is a document that I need to store in local memory.
        istringstream ss(line);
        string word;
        int count;

        // Adjust initial topic probabilities
        vector<int32> label_topics;
        vector<int32> candidate_topics;
      //  for (int i = 0; i < num_topics; ++i) {
      //    candidate_topics.push_back(i);
      //  }
        if (line[0] == '[') {
          while (ss >> word) {  
            if (word[0] == '[')
              word = word.substr(1, word.length());
            if (word[word.length()-1] == ']') {
              word = word.substr(0, word.length()-1);
              //printf("%s\n", word.c_str());
              label_topics.push_back(atoi(word.c_str()));
              break;
            } else {
              label_topics.push_back(atoi(word.c_str()));
            }
          }
        }

        // Init word assignments
        DocumentWordTopicsPB document;
        set<string> words_in_document;
        while (ss >> word >> count) {
          if (word_index_map.end() == word_index_map.find(word))
            continue;
          int word_index = word_index_map.find(word)->second;

          vector<int32> topics;
          // seed word
          if (word_topics_map.end() != word_topics_map.find(word_index)) {
            vector<int32> seed_topics = word_topics_map.find(word_index)->second;
            for (int i = 0; i < count; ++i) {
              topics.push_back(
                  seed_topics[RandInt(seed_topics.size())] );
              candidate_topics.push_back(
                  seed_topics[RandInt(seed_topics.size())] );
            }
          // labeled document
          } else if (label_topics.size() > 0) {
            for (int i = 0; i < count; ++i) {
              topics.push_back(
                  label_topics[RandInt(label_topics.size())] );
            }
          // contains seed words
          } else if (candidate_topics.size() > 0) {
            for (int i = 0; i < count; ++i) {
              topics.push_back(
                  candidate_topics[RandInt(candidate_topics.size())] );
            }
          // randomize
          } else {
            for (int i = 0; i < count; ++i) {
              topics.push_back(
                  RandInt(num_topics) );
            }
          }
          document.add_wordtopics(word, word_index, topics);
          words_in_document.insert(word);
        }
        if (words_in_document.size() > 0) {
          corpus->push_back(new LDADocument(document, num_topics, label_topics));
        }
      }
      index++;
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
}

int main(int argc, char** argv) {
  using learning_lda::LoadWordIndex;
  using learning_lda::LoadSeedWords;
  using learning_lda::LDACorpus;
  using learning_lda::LDAModel;
  using learning_lda::ParallelLDAModel;
  using learning_lda::LDASampler;
  using learning_lda::DistributelyLoadAndInitTrainingCorpus;
  using learning_lda::LDACmdLineFlags;

  int myid, pnum;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Comm_size(MPI_COMM_WORLD, &pnum);

  LDACmdLineFlags flags;
  flags.ParseCmdFlags(argc, argv);
  if (!flags.CheckParallelTrainingValidity()) {
    return -1;
  }

  map<string, int> word_index_map;
  if (!flags.init_model_file_.empty()) {
    ifstream model_fin(flags.init_model_file_.c_str());
    LDAModel old_model(model_fin, &word_index_map);
    if (myid == 0) {
      std::cout << "after init model, word_index_map size: " << int(word_index_map.size()) << std::endl;
    }
  } else if (!flags.word_index_file_.empty()) {
    int n_words = LoadWordIndex(flags.word_index_file_, word_index_map);
    if (myid == 0) {
      std::cout<<"word_index_map size: "<<n_words<<std::endl;
    }
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
  CHECK_GT(DistributelyLoadAndInitTrainingCorpus(flags.training_data_file_,
                                     flags.num_topics_,
                                     myid, pnum, 
                                     word_index_map,
                                     word_topics_map,
                                     &corpus), 0);
  std::cout << "Training data loaded" << std::endl;
  
  if (!flags.init_model_file_.empty()) {
    ifstream model_fin(flags.init_model_file_.c_str());
    LDAModel old_model(model_fin, &word_index_map);
    LDASampler old_sampler(flags.alpha_, flags.beta_, &old_model, NULL, &word_topics_map);
    for (int iter=0; iter<flags.burn_in_iterations_; ++iter) {
      if (myid == 0) {
        std::cout << "Iteration using old model " << iter << " ...\n";
      }
      old_sampler.DoIteration(&corpus, false, iter<flags.burn_in_iterations_);
    }
  }

  for (int iter = 0; iter < flags.total_iterations_; ++iter) {
    if (myid == 0) {
      std::cout << "Iteration " << iter << " ...\n";
    }

    ParallelLDAModel model(flags.num_topics_, word_index_map);
    model.ComputeAndAllReduce(corpus);
    LDASampler sampler(flags.alpha_, flags.beta_, 
                       &model, NULL, &word_topics_map);

    if (flags.compute_likelihood_ == "true" &&
        iter % 5 == 0) {
      double loglikelihood_local = 0;
      double loglikelihood_global = 0;
      for (list<LDADocument*>::const_iterator iter = corpus.begin();
           iter != corpus.end();
           ++iter) {
        loglikelihood_local += sampler.LogLikelihood(*iter);
      }
      MPI_Allreduce(&loglikelihood_local, 
                    &loglikelihood_global, 
                    1, 
                    MPI_DOUBLE,
                    MPI_SUM, 
                    MPI_COMM_WORLD);
      if (myid == 0) {
        std::cout << "Loglikelihood: " << loglikelihood_global << std::endl;
      }
    }

    sampler.DoIteration(&corpus, true, false);
  }

  ParallelLDAModel model(flags.num_topics_, word_index_map);
  model.ComputeAndAllReduce(corpus);
  if (myid == 0) {
    std::ofstream fout(flags.model_file_.c_str());
    model.AppendAsString(fout);
  }
  FreeCorpus(&corpus);
  MPI_Finalize();
  return 0;
}


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

#include <math.h>
#include <stdlib.h>

#include "sampler.h"
#include "document.h"
#include "model.h"

namespace learning_lda {

LDASampler::LDASampler(double alpha,
                       double beta,
                       LDAModel* model,
                       LDAAccumulativeModel* accum_model,
                       map<int, vector<int32> >* word_topics_map)
    : alpha_(alpha), beta_(beta), model_(model), accum_model_(accum_model), word_topics_map_(word_topics_map) {
  CHECK_LT(0.0, alpha);
  CHECK_LT(0.0, beta);
  CHECK(model != NULL);
}

void LDASampler::InitModelGivenTopics(const LDACorpus& corpus) {
  for (list<LDADocument*>::const_iterator iter = corpus.begin();
       iter != corpus.end();
       ++iter) {
    LDADocument* document = *iter;
    for (LDADocument::WordOccurrenceIterator iter2(document);
         !iter2.Done();
         iter2.Next()) {
      model_->IncrementTopic(iter2.Word(), iter2.Topic(), 1);
    }
  }
}

void LDASampler::DoIteration(LDACorpus* corpus,
    bool train_model, bool burn_in) {
  for (list<LDADocument*>::iterator iter = corpus->begin();
       iter != corpus->end();
       ++iter) {
    SampleNewTopicsForDocument(*iter, train_model);
  }
  if (accum_model_ != NULL && train_model && !burn_in) {
    accum_model_->AccumulateModel(*model_);
  }
}

void LDASampler::SampleNewTopicsForDocument(LDADocument* document,
    bool update_model) {
  for (LDADocument::WordOccurrenceIterator iterator(document);
       !iterator.Done();
       iterator.Next()) {
    // This is a (non-normalized) probability distribution from which we will
    // select the new topic for the current word occurrence.
    CHECK(NULL != word_topics_map_);
    vector<double> new_topic_distribution;

    vector<int32>* seed_topics = NULL;
    if (word_topics_map_->end() != word_topics_map_->find(iterator.Word())) {
      seed_topics = &(word_topics_map_->find(iterator.Word())->second);
    }
    vector<int32>* label_topics = NULL;
    if (document->label_topics_.size() > 0) {
      label_topics = &(document->label_topics_);
    }
    GenerateTopicDistributionForWord(*document,
                                     iterator.Word(),
                                     iterator.Topic(),
                                     update_model,
                                     seed_topics,
                                     label_topics,
                                     &new_topic_distribution);
    int new_topic = GetAccumulativeSample(new_topic_distribution);
    // Update document and model parameters with the new topic.
    if (update_model) {
      model_->ReassignTopic(
          iterator.Word(), iterator.Topic(), new_topic, 1);
    }
    iterator.SetTopic(new_topic);
  }
}

void LDASampler::GenerateTopicDistributionForWord(
    const LDADocument& document,
    int word,
    int current_word_topic,
    bool train_model,
    const vector<int32>* seed_topics, 
    const vector<int32>* label_topics,
    vector<double>* distribution) const {
  int num_topics = model_->num_topics();
  CHECK_LT(0, num_topics);
  int num_words = model_->num_words();
  distribution->clear();
  distribution->reserve(num_topics);

  const TopicCountDistribution& word_distribution =
      model_->GetWordTopicDistribution(word);
  for (int k = 0; k < num_topics; ++k) {
    // We will need to temporarily unassign the word from its old topic, which
    // we accomplish by decrementing the appropriate counts by 1.
    int current_topic_adjustment = (train_model && k == current_word_topic) ? -1 : 0;

    double topic_word_factor = 
        word_distribution[k] + current_topic_adjustment;

    double global_topic_factor =
        model_->GetGlobalTopicDistribution()[k] + current_topic_adjustment;

    double document_topic_factor =
        document.topic_distribution()[k] + current_topic_adjustment;

    distribution->push_back(
        (topic_word_factor + beta_) *
        (document_topic_factor + alpha_) /
        (global_topic_factor + num_words * beta_));
  }
  if (NULL != seed_topics) {
    for (int k = 0; k < seed_topics->size(); ++k) {
      int index = (*seed_topics)[k];
      (*distribution)[index] *= 20;
    }
  }
  if (NULL != label_topics) {
    for (int k = 0; k < label_topics->size(); ++k) {
      int index = (*label_topics)[k];
      (*distribution)[index] *= 10;
    }
  }
}

// Compute log P(d) = sum_w log P(w), where P(w) = sum_z P(w|z)P(z|d).
double LDASampler::LogLikelihood(LDADocument* document) const {
  const int num_topics(model_->num_topics());

  // Compute P(z|d) for the given document and all topics.
  const vector<int64>& document_topic_cooccurrences(
      document->topic_distribution());
  CHECK_EQ(num_topics, document_topic_cooccurrences.size());
  int64 document_length = 0;
  for (int t = 0; t < num_topics; ++t) {
    document_length += document_topic_cooccurrences[t];
  }
  vector<double> prob_topic_given_document(num_topics);
  for (int t = 0; t < num_topics; ++t) {
    prob_topic_given_document[t] =
        (document_topic_cooccurrences[t] + alpha_) /
        (document_length + alpha_ * num_topics);
  }

  // Get global topic occurrences, which will be used compute P(w|z).
  TopicCountDistribution global_topic_occurrences(
      model_->GetGlobalTopicDistribution());

  double log_likelihood = 0.0;

  // A document's likelihood is the product of its words' likelihoods.  Compute
  // the likelihood for every word and sum the logs.
  for (LDADocument::WordOccurrenceIterator iterator(document);
       !iterator.Done();
       iterator.Next()) {
    // Get topic_count_distribution of the current word, which will be
    // used to Compute P(w|z).
    TopicCountDistribution word_topic_cooccurrences(
        model_->GetWordTopicDistribution(iterator.Word()));

    // Comput P(w|z).
    vector<double> prob_word_given_topic(num_topics);
    for (int t = 0; t < num_topics; ++t) {
      prob_word_given_topic[t] =
          (word_topic_cooccurrences[t] + beta_) /
          (global_topic_occurrences[t] + model_->num_words() * beta_);
    }

    // Compute P(w) = sum_z P(w|z)P(z|d)
    double prob_word = 0.0;
    for (int t = 0; t < num_topics; ++t) {
      prob_word += prob_word_given_topic[t] * prob_topic_given_document[t];
    }

    log_likelihood += log(prob_word);
  }
  return log_likelihood;
}

}  // namespace learning_lda

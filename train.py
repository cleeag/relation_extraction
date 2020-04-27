import os
from os.path import join
import json

from allennlp.data.vocabulary import Vocabulary

from tre.dataset_readers import open_nre_nyt_reader
from tre.byte_pair_indexer import OpenaiTransformerBytePairIndexer
from tre import bag_iterator
from tre.model import MITRE


def run():
    config = json.load(open("/home/cleeag/relation_extraction/experiments/configs/model_paper.json", 'r'))
    reader_params = config['dataset_reader']
    iterator_params = config['iterator']
    trainer_params = config['trainer']
    vocab = Vocabulary()
    label_encoder = json.loads(open("/home/cleeag/relation_extraction/data/open_nre_nyt/rel2id.json", 'r').read())
    for word, idx in label_encoder.items():
        vocab._token_to_index['labels'][word] = idx
        vocab._index_to_token['labels'][idx] = word

    tre_byte_pair_indexer = OpenaiTransformerBytePairIndexer(
        model_path=reader_params['token_indexers']['byte_pairs']['model_path'],
        tokens_to_add=reader_params['token_indexers']['byte_pairs']['tokens_to_add'],
        vocabulary=vocab)

    data_reader = open_nre_nyt_reader.OpenNreNYTReader("ner_most_specific", {"byte_pairs": tre_byte_pair_indexer})
    dataset = data_reader.read("/home/cleeag/relation_extraction/data/open_nre_nyt/small_test.json")
    batch_iterator = bag_iterator.BagIterator(sorting_keys=iterator_params["sorting_keys"],
                                              biggest_batch_first=iterator_params['biggest_batch_first'],
                                              batch_size=iterator_params['batch_size'],
                                              maximum_samples_per_batch=iterator_params['maximum_samples_per_batch'],
                                              vocab=vocab)


    # vocab.add_tokens_to_namespace(reader_params['token_indexers']['tokens_to_add'])
    vocab.print_statistics()

    # model = MITRE(vocab,
    #              openai_model_path: ,
    #              n_ctx: int=512,
    #              tokens_to_add: List[str]=None,
    #              requires_grad: bool=True,
    #              clf_token: str='__clf__',
    #              dropout: float=.1,
    #              entity_dropout: float=0.0,
    #              language_model_weight: float=.5,
    #              selector: str='average',
    #              label_namespace='labels')

    for batch in batch_iterator(dataset, num_epochs=trainer_params['num_epochs']):
        pass


if __name__ == '__main__':
    run()

# -*- coding: utf-8 -*-

from flask import Flask, jsonify
from flask_restful import reqparse, Api, Resource
from nlpir_entity import nlpir
from stat_freq import build_test_data_from_crf
from cut_freq import build_test_data_from_crf_cut
from ltp_parser import parse
from ltp_parser import segmentor_tag
from mlstm_sentiment import predict_model
from text_similar import cal_similarity
from w2v import word2vec_str
from classifier_text import  text_classifier

app = Flask(__name__)
api = Api(app)


class entity(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('content', type=str, required=True)
        args = parser.parse_args()

        try:
            sset = nlpir(args['content'])
            result_list = []
            for disease, couple in sset.items():
                single_dic = {}
                single_dic['disease'] = disease
                single_dic['symptom'], single_dic['symptomAll'] = couple[0], couple[1]
                result_list.append(single_dic)
            return result_list
        except:
            return jsonify({'message': 'unknown error'})


class freq(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('content', type=str)
        content = parser.parse_args()['content']
        freq_result = build_test_data_from_crf(content)
        freq_result_dic = {}
        result = {}
        result_n = []
        result_v = []
        result_adj = []
        result_symptom = []
        for i in freq_result:
            for word, count in i.items():
                freq_result_dic[word] = count

        for word, count in freq_result_dic.items():
            if word == '':
                pass
            else:
                a = build_test_data_from_crf_cut(word)
                type = a[0]['type']
                if type == 'n':
                    result_n.append({'word': word, 'count': count})

                elif type == 'adj':
                    result_adj.append({'word': word, 'count': count})

                elif type == 'v':
                    result_v.append({'word': word, 'count': count})

                elif type == 'symptom':
                    result_symptom.append({'word': word, 'count': count})

        result['n'] = result_n
        result['v'] = result_v
        result['adj'] = result_adj
        result['symptom'] = result_symptom

        return result

class parser_ltp(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('content', type=str)
        content = parser.parse_args()['content']
        return parse(content)

class segmentor(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('content', type=str)
        content = parser.parse_args()['content']
        words_list, tags_list, symptom_flag_list = segmentor_tag(content)
        segment_result = []
        id = 1
        for word,tag,symptom_flag in zip(words_list,tags_list,symptom_flag_list):
            a = {}
            a['id'] = id
            a['word'] = word
            a ['tag'] = tag
            a['symptom_flag'] = symptom_flag
            segment_result.append(a)
            id += 1
        return segment_result


class sentiment(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('content', type=str)
        content = parser.parse_args()['content']
        result = predict_model(content)

        result_list = []
        single_dic = {}
        for i, j in result.items():
            single_dic.update(j)
        for word, value in single_dic.items():
            result_dic = {}
            result_dic['word'] = word
            result_dic['value'] = value
            result_list.append(result_dic)

        return jsonify(result_list)


class text_similar(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('content1', type=str)
        parser.add_argument('content2', type=str)
        parser.add_argument('type', type=int)

        content1 = parser.parse_args()['content1']
        content2 = parser.parse_args()['content2']
        type = parser.parse_args()['type']

        result = cal_similarity(content1, content2, type)
        return result


# type = 8 -- 计算短文本相似度
# type = 9 -- 计算语义相似度

class w2v(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('content', type=str)
        content = parser.parse_args()['content']
        result = word2vec_str(content)
        result_list = []
        for i, j in result.items():
            for word, value in j.items():
                single_dic = {}
                single_dic['word'] = word
                single_dic['value'] = value
            result_list.append(single_dic)
        return jsonify(result_list)


class classifier(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('content', type=str)
        content = parser.parse_args()['content']

        try:

            # print(result)
            # return jsonify({'state': '0'})
            result = text_classifier(content)
            return result
        except:
            return jsonify({'state': '1'})


api.add_resource(classifier, '/classifier')
api.add_resource(w2v, '/w2v')
api.add_resource(text_similar, '/text_similar')
api.add_resource(sentiment, '/sentiment')
api.add_resource(parser_ltp, '/parser')
api.add_resource(segmentor,'/segmentor')
api.add_resource(freq, '/freq')
api.add_resource(entity, '/entity')

if __name__ == '__main__':
    app.run(debug=True, port=5555)

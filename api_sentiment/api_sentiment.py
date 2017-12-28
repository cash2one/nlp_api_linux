# -*- coding:utf-8 -*-

from flask import Flask, jsonify
from flask_restful import Resource, Api, reqparse
from mlstm_sentiment import predict_model


app = Flask(__name__)
api = Api(app)



class sentiment(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('content', type=str)
        content = parser.parse_args()['content']
        result = predict_model(content)
        
        result_list = []
        single_dic = {}
        for i,j in result.items():
            single_dic.update(j)
        for word,value in single_dic.items():
            result_dic = {}
            result_dic['word'] = word
            result_dic['value'] = value
            result_list.append(result_dic)
    
        return jsonify(result_list)

api.add_resource(sentiment, '/sentiment')


if __name__ == '__main__':
    app.run(port=5558,debug=True)
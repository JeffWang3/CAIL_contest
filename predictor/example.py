from predictor import Predictor


content = "公诉机关指控被告人王彬县名仕花园经营一家足浴店于年月日凌晨时许年月日时许年月日晚时许年月日凌晨时许应杨王王某党刘某刘将梁"
content += "某梁刘某杨送杨王王某党刘某刘彬县南街商务宾馆号房间彬县金海岸宾馆房间彬县和谐宾馆彬县长安宾馆房间收取现金元元元元公诉机关"
content +="指控提供书证证人证言被告人供述辩解证据证实被告人王介绍容留妇"
content +="女卖淫罪请求依法判处"


pre = Predictor()
print(pre.predict([content]))

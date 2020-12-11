# -*- encoding: utf-8 -*-
'''
LinearRegressionUtil.py
Created on 2020/11/19 19:40
Copyright (c) 2020/11/19, Google Copy right
@author: 梁吉
'''

import torch
import matplotlib.pyplot as plt
from numpy import *
from collections import defaultdict
import operator
import jaydebeapi


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden,n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = self.hidden(x)
        x = torch.tanh(x)
        x = self.predict(x)
        return x

"""
    封装原始数据=》[(x,y),(x,y),(x,y)]
"""
def lineToTupleList(tup):
    tupleList=[]
    for i in tup:
        str = i[11:len(i)-1]
        list = str.split(',')       # 从LINESTRING截出每个(x y)坐标点
        for j in list:
            lonlat = j.split(' ')
            tupleList.append((float(lonlat[0]),float(lonlat[1])))       # 封装成二元组，所有的都装进一个大List
    return tupleList


"""
    封装原始数据=》[(y,x),(y,x),(y,x)]
"""
def lineToTupleListTURN(tup):
    tupleList=[]
    for i in tup:
        str = i[11:len(i)-1]
        list = str.split(',')       # 从LINESTRING截出每个(x y)坐标点
        for j in list:
            lonlat = j.split(' ')
            tupleList.append((float(lonlat[1]),float(lonlat[0])))       # 封装成二元组，所有的都装进一个大List
    return tupleList


"""
    list<tuple> ==》 [listX[],listY[]]
"""
def tupleListToListXY(tup):
    listX = []
    listY = []
    for i in tup:
        listX.append(i[0])
        listY.append(i[1])
    return [listX, listY]

"""
    返回一个List内重复元素下标
"""
def list_duplicates(sourceList):
    repetIndexList=[]
    tally = defaultdict(list)
    for i,item in enumerate(sourceList):
        tally[item].append(i)
    tmp= ((key,locs) for key,locs in tally.items() if len(locs)>1)
    for dup in sorted(tmp):
        repetIndexList=repetIndexList+dup[1]
    return repetIndexList

"""
 判断是否需要翻转x、y        （A港经度，A港纬度，B港经度，B港维度，AIS数据）
 return  0-翻转；1-不翻
 """
def ifTurn(portALot,portALat,portBLot,portBLat,listX,listY):
    sumLotOverNumber=sum(i > max(portALot,portBLot) or i < min(portALot,portBLot) for i in listX)      # 经度超范围的总个数
    lotOverPrecent=sumLotOverNumber/len(listX)  # 经度超过的百分比
    if(lotOverPrecent > 0.1):
        sumLatOverNumber=sum(i > max(portALat,portBLat) or i < min(portALat,portBLat) for i in listY)   # 如果经度超了再判断纬度超否
        latOverPrecent=sumLatOverNumber/len(listY)  # 纬度超过的百分比
        if( latOverPrecent> 0.1):
            print("经纬度超过阈值，可能无法拟合")
            print("经度、纬度所超百分比 %f %f" %(sumLotOverNumber/len(listX),sumLatOverNumber/len(listY)))
            if(lotOverPrecent>latOverPrecent):  # 虽然经纬度都超了，总得取一个最优的计算
                return 0
            else:
                return 1
        else:
            print("经度超了，维度没超，开始翻转")     # 经度超 纬度没超
            print("经度所超百分比 %f" %(sumLotOverNumber/len(listX)))
            return 0
    else:
        print("经纬度都没超，最好不过了")             # 经纬度都没超
        return 1

"""
    如果过段过陆地总长超过20km才算过陆地，否则算过暗礁，该段不要再拟合
"""
def isCrossLand(line,threshold,curs):
    threshold = threshold/100
    """ 1、先找过陆地的多边形 2、循环判断过陆地的最大长度 """
    sql1 = "SELECT ST_ASTEXT(GEOM) FROM world_country WHERE geom && st_geomfromtext('"+line+"') AND st_intersects(st_geomfromtext('"+line+"'),geom)"
    curs.execute(sql1)
    result1 = curs.fetchall()    ## list(tuple)
    for row in result1:
        ## 查询与陆地相交
        sql2 = "SELECT ST_Length(ST_Intersection(ST_Geomfromtext('" + line + "'),ST_Geomfromtext('" + row[0] + "')))"
        # sql2 = "SELECT ST_Intersection(ST_Geomfromtext('" + line + "'),ST_Geomfromtext('" + row[0] + "'))"
        curs.execute(sql2)
        result2 = curs.fetchall()
        return threshold < result2[0][0]
    return False



def linearRegress(portA_lot, portA_lat, portB_lot, portB_lat, lineList):
    tup = lineList

    # 封装成二元组List List<Tuple(x,y)>=====
    myXYTupleList=lineToTupleList(tup)

    # 判断是否翻转
    listXY = tupleListToListXY(myXYTupleList)
    listX=listXY[0]   # [103.76751, 103.767518333, ......, 103.767583333,]
    listY=listXY[1]   # [1.26997833333, 1.26995, 1.26991166667, ......, 1.27000833333]
    tag=1
    tag=ifTurn(portA_lot,portA_lat,portB_lot,portB_lat,listX,listY)

    # 如果翻转了，直接对调xy，后面都不需要分类讨论了不是吗
    if(tag==0):
        myXYTupleList=lineToTupleListTURN(tup)

    # 点排序默认经度排序（否则多条线）=====
    myXYTupleList.sort(key= operator.itemgetter(0))

    # 排序后拆开去重=====
    listXY = tupleListToListXY(myXYTupleList)
    listX=listXY[0]   # [103.76751, 103.767518333, ......, 103.767583333,]
    listY=listXY[1]   # [1.26997833333, 1.26995, 1.26991166667, ......, 1.27000833333]
    print("原listX、listY长度为：%d %d" %(len(listX),len(listY)))
    listXRepetList=list_duplicates(listX)   # listX中重复元素下标
    listX = [listX[i] for i in range(len(listX)) if (i not in listXRepetList)]
    listY = [listY[i] for i in range(len(listY)) if (i not in listXRepetList)]
    listYRepetList=list_duplicates(listY)   # listY中重复元素下标
    listX = [listX[i] for i in range(len(listX)) if (i not in listYRepetList)]
    listY = [listY[i] for i in range(len(listY)) if (i not in listYRepetList)]
    print("去重后listX、listY为：%d %d" %(len(listX),len(listY)))

    # 归一化=====
    meanX=mean(listX)
    meanY=mean(listY)
    listX=[x-meanX for x in listX]
    listY=[y-meanY for y in listY]

    # 格式转化=====
    x =torch.tensor(listX, dtype=torch.float)
    y =torch.tensor(listY, dtype=torch.float)
    x=torch.unsqueeze(x,1)
    y=torch.unsqueeze(y,1)

    # 定义初始变量
    net = Net(n_feature=1, n_hidden=20,n_output=1)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.1)        # optim是优化器，用来初始所有w权重参数。lr是学习率，一般设置小于1。
    loss_func = torch.nn.MSELoss()           # MSELoss损失函数，这里用均方误差，但吴恩达说这找不到全局最优解
    bestLoss=999            # 用来保存最好的一次loss
    bestPrediction=net(x)   # 用来保存最好的一次预测值



    # 初始化h2连接，用于判断点是否在陆地上
    dirver = 'org.h2.Driver'
    url = 'jdbc:h2:tcp://192.168.0.242:9101/~/ship5'
    username = 'sa'
    password = ''
    # jar = 'D:/development/h2gis-standalone/h2gis-dist-1.5.0.jar'
    jar = 'D:/Hadoop/H2/h2gis-standalone/h2gis-dist-1.5.0.jar'
    # jar = '/usr/local/hadoop/h2gis-standalone/h2gis-dist-1.5.0.jar'
    conn = jaydebeapi.connect(dirver, url, [username, password], jar)
    curs = conn.cursor()


    # 开始正式训练
    for t in range(2000):   # 训练多少次，应当为2000次
        prediction = net(x)
        loss = loss_func(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 取Loss最小也就是最好的一次存下来
        if(loss.data.numpy()<bestLoss):
            bestLoss=loss.data.numpy()
            bestPrediction=prediction

    # 最好一次拟合的bestPrediction   由tensor转为pythonList  留后面各个过陆地段单独拟合后替换进去。
    row=bestPrediction.data.numpy().shape[0]    # numpy的行数,作为元素的个数
    xListFromNumpy=x.data.numpy().tolist()
    bestPredictionListFromNumpy=bestPrediction.data.numpy().tolist()  # numpy转list，只不过转的list时[[number],[number]]


    # 打印第一次拟合的曲线，顺便计算拟合线长度判断要切几片
    sectionNumber=8
    if(tag==0):
        LINESTRING=""
        for i in range(row):
            pointStr=str(bestPredictionListFromNumpy[i][0]+meanY)+" "+str(xListFromNumpy[i][0]+meanX)
            LINESTRING+=(pointStr)+","
        LINESTRING=LINESTRING[0:len(LINESTRING)-1]  # 去最后一个逗号
        LINESTRING="LINESTRING("+LINESTRING+")"
        print("未修正前拟合航迹线为：（也就是会过陆地的拟合线）")
        print(LINESTRING)
        if(len(LINESTRING) < 100): return LINESTRING, -1
        sql = "SELECT ST_Length('"+LINESTRING+"')"
        curs.execute(sql)
        result = curs.fetchall()[0][0]
        sectionNumber=int(result/10)
        if sectionNumber==0:
            sectionNumber=1
        print("sectionNumber切片数为：%d" %sectionNumber )
    else:
        LINESTRING=""
        for i in range(row):
            pointStr=str(xListFromNumpy[i][0]+meanX)+" "+str(bestPredictionListFromNumpy[i][0]+meanY)
            LINESTRING+=(pointStr)+","
        LINESTRING=LINESTRING[0:len(LINESTRING)-1]  # 去最后一个逗号
        LINESTRING="LINESTRING("+LINESTRING+")"
        print("未修正前拟合航迹线为：")
        print(LINESTRING)
        if (len(LINESTRING) < 100): return LINESTRING, -1
        sql = "SELECT ST_Length('"+LINESTRING+"')"
        curs.execute(sql)
        result = curs.fetchall()[0][0]
        sectionNumber=int(result/10)
        if sectionNumber==0:
            sectionNumber=1
        print("sectionNumber切片数为：%d" %sectionNumber )


    # 找拟合线哪几段过了陆地。overLandList存的是每一段的start-end，是start-end，是start-end！！！！！！！！
    overLandList=[]     # 二元组形式存拟合线每段过陆地的start-end索引。[tuple(start,end)]，拿着start和end去listSource取出start~end内所有点。
    numPart=0
    start=0
    end=int(row/sectionNumber)  # 大不了100等分
    eachPart=int(row/sectionNumber)
    if(tag==0):
        for j in range(sectionNumber):
            LINESTRING=""
            for i in range(start,end):          #把拟合线多等分，单个段判断是否与陆地相交
                pointStr=str(bestPredictionListFromNumpy[i][0]+meanY)+" "+str(xListFromNumpy[i][0]+meanX)
                LINESTRING+=(pointStr)+","
            LINESTRING=LINESTRING[0:len(LINESTRING)-1]  # 去最后一个逗号
            LINESTRING="LINESTRING("+LINESTRING+")"
            sql = "SELECT gid FROM world_country WHERE (geom && st_geomfromtext('"+LINESTRING+"')) AND st_crosses(st_geomfromtext('"+LINESTRING+"'), geom) limit 1"
            curs.execute(sql)
            result = curs.fetchall()
            if(len(result)>0):
                landOverThresholdTag=isCrossLand(LINESTRING,15,curs)     # 如果该段超过阈值，才算真正过陆地，才要再拟合
                if landOverThresholdTag==True:
                    overLandList.append((start,end))
                    print("第 %d 段过陆地的原始拟合线为：" %j)
                    print(LINESTRING)
                else:
                    print("没超过阈值，放弃拟合")
            numPart=numPart+1
            LINESTRING=""   # 重置
            start=end
            end=end+eachPart
        print("拟合线过陆地的点的start-end索引为：")
        print(overLandList)    # overLandList[0][0]就是第一段过陆地的start，overLandList[0][1]就是第一段过陆地的end；overLandList[1][0]就是第二段过陆地的start
    else:
        for j in range(sectionNumber):
            LINESTRING=""
            for i in range(start,end):          #把拟合线多等分，单个段判断是否与陆地相交
                pointStr=str(xListFromNumpy[i][0]+meanX)+" "+str(bestPredictionListFromNumpy[i][0]+meanY)
                LINESTRING+=(pointStr)+","
            LINESTRING=LINESTRING[0:len(LINESTRING)-1]  # 去最后一个逗号
            LINESTRING="LINESTRING("+LINESTRING+")"
            sql = "SELECT gid FROM world_country WHERE (geom && st_geomfromtext('"+LINESTRING+"')) AND st_crosses(st_geomfromtext('"+LINESTRING+"'), geom) limit 1"
            curs.execute(sql)
            result = curs.fetchall()
            if(len(result)>0):
                landOverThresholdTag=isCrossLand(LINESTRING,15,curs)     # 如果该段超过阈值，才算真正过陆地，才要再拟合
                if landOverThresholdTag==True:
                    overLandList.append((start,end))
                    print("第 %d 段过陆地的原始拟合线为：" %j)
                    print(LINESTRING)
                else:
                    print("没超过阈值，放弃拟合")
            else:
                print("第一次拟合就没过陆地，直接返回")
                return LINESTRING, bestLoss
            numPart=numPart+1
            LINESTRING=""   # 重置
            start=end
            end=end+eachPart
        print("拟合线过陆地的点的start-end索引为：")
        print(overLandList)    # overLandList[0][0]就是第一段过陆地的start，overLandList[0][1]就是第一段过陆地的end；overLandList[1][0]就是第二段过陆地的start


    # 对于过陆地的每一段，找到原始数据单独再拟合
    for i in overLandList:
        start=i[0]      # 拟合线每段过陆地的起始index，起始也就是原始数据的起始index
        end=i[1]
        # 从原始数据取出这一段的原始数据，这是已经归一化后的。在最后一步转LINESTRING时再统一归一化?不行！需要再归一化然后再反归一化
        rangListX=listX[start:end]
        rangListY=listY[start:end]      # rangListY下标从0开始
        # 小范围也判断翻不翻转？？  因为原始数据已经排序了，所以该小范围不好判断翻不翻转因为listX[start],listY[start],listX[end],listY[end]已经是错的永远是没超不用翻转
        # rangeTag=ifTurn(listX[start],listY[start],listX[end],listY[end],rangListX,rangListY)
        # if rangeTag==0:
        #     print("该小范围要翻转")
        # 小范围也要再归一化
        meanRangeX=mean(rangListX)
        meanRangeY=mean(rangListY)
        rangListX=[x-meanRangeX for x in rangListX]
        rangListY=[y-meanRangeY for y in rangListY]
        # 格式转化=====
        rangeX =torch.tensor(rangListX, dtype=torch.float)
        rangeY =torch.tensor(rangListY, dtype=torch.float)
        rangeX=torch.unsqueeze(rangeX,1)
        rangeY=torch.unsqueeze(rangeY,1)

        rangeBestLoss=999           # 小范围再拟合用来保存最好的一次loss
        rangeBestPrediction=net(rangeX)
        # 开始该小范围的训练迭代
        for t in range(2000):   # 训练多少次，应当为2000次
            rangePrediction = net(rangeX)
            loss = loss_func(rangePrediction, rangeY)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 取Loss最小也就是最好的一次存下来
            if(loss.data.numpy()<rangeBestLoss):
                rangeBestLoss=loss.data.numpy()
                rangeBestPrediction=rangePrediction
        # # 需要对各个段反归一化。然后拼回到bestPredictionListFromNumpy。最终后面整体加上meanX、meanY
        rangePredictionListFromNumpy=rangeBestPrediction.data.numpy().tolist()
        tmp=start
        for i in range(end-start):  # rangePrediction下标也是从0开始的
            bestPredictionListFromNumpy[tmp][0]=rangePredictionListFromNumpy[i][0]+meanRangeY
            tmp=tmp+1


    # 阶梯段修正    交界段左右处各取出1%的数据。
    # 下面这个是补全overlandList==>overLandListCompletion  直接连接overlandList中的start-end-start-end...会丢失没过陆地的拟合线
    overLandListCompletion=[]
    lastStop=0
    for i in range(len(overLandList)):
        partstart=overLandList[i][0]    # 拿到每一段的开头索引
        partend=overLandList[i][1]
        if partstart!=lastStop:
            startAndEndIndexTuple=(lastStop,partstart)
            overLandListCompletion.append(startAndEndIndexTuple)    # 把空缺段补进去
        overLandListCompletion.append(overLandList[i])      # 把这次过陆地的加进去
        lastStop=partend
        if i==(len(overLandList)-1): # 过陆地段最后个时，判断后面有没有要补全的了
            if row-partend>(eachPart*0.9):    #最后一个过陆地段后面还有没补的
                startAndEndIndexTuple=(partend,row)
                overLandListCompletion.append(startAndEndIndexTuple)
    print("过陆地段补全后的overLandListCompletion为：")
    print(overLandListCompletion)
    # 想法是overLandListCompletion每个间隔中去掉1%的点
    ignoreNumber=int(row*0.01)
    afterIgnoredXList=[]
    afterIgnoredPredictionList=[]
    for i in range(len(overLandListCompletion)):
        partstart=overLandListCompletion[i][0]    # 拿到每一段的开头索引
        partend=overLandListCompletion[i][1]      # 拿到每一段的结束索引
        if partstart==0 and row-partend<eachPart:            # 拟合线只有一段并且还过陆地了，不用ignore了直接拼好break掉
            afterIgnoredXList[0:]=xListFromNumpy[0:partend]
            afterIgnoredPredictionList[0:]=bestPredictionListFromNumpy[0:partend]
            break
        if partstart==0 and row-partend>eachPart:            # 过陆地的是第一段但不是最后一段，该段起始处就不需要+ignoreNumber，结尾处需要-ignoreNumber
            afterIgnoredXList[len(afterIgnoredXList):]=xListFromNumpy[0:partend-ignoreNumber]
            afterIgnoredPredictionList[len(afterIgnoredPredictionList):]=bestPredictionListFromNumpy[0:partend-ignoreNumber]
        if partstart!=0 and row-partend<eachPart:    # 过陆地的不是第一段但是最后一段，该段起始处要+ignoreNumber，结尾处不需要-ignoreNumber
            afterIgnoredXList[len(afterIgnoredXList):]=xListFromNumpy[partstart+ignoreNumber:]
            afterIgnoredPredictionList[len(afterIgnoredPredictionList):]=bestPredictionListFromNumpy[partstart+ignoreNumber:]
        if partstart!=0 and row-partend>eachPart:    # 既不是第一段也不是最后一段
            if len(afterIgnoredXList)==0:   # 如果不加，第一段就会丢失
                afterIgnoredXList[0:]=xListFromNumpy[0:partstart-ignoreNumber]
                afterIgnoredPredictionList[0:]=bestPredictionListFromNumpy[0:partstart-ignoreNumber]
            afterIgnoredXList[len(afterIgnoredXList):]=xListFromNumpy[partstart+ignoreNumber:partend-ignoreNumber]
            afterIgnoredPredictionList[len(afterIgnoredPredictionList):]=bestPredictionListFromNumpy[partstart+ignoreNumber:partend-ignoreNumber]


    # 打印未修正前LINESTRING，只为做比较，不用输出到数据库
    if(tag==0):
        LINESTRING=""
        for i in range(row):
            pointStr=str(bestPredictionListFromNumpy[i][0]+meanY)+" "+str(xListFromNumpy[i][0]+meanX)
            LINESTRING+=(pointStr)+","
        LINESTRING=LINESTRING[0:len(LINESTRING)-1]  # 去最后一个逗号
        LINESTRING="LINESTRING("+LINESTRING+")"
        print("陆地边界修正之后的拟合线为：（也就是有台阶的拟合线）")
        print(LINESTRING)
    else:
        LINESTRING=""
        for i in range(row):
            pointStr=str(xListFromNumpy[i][0]+meanX)+" "+str(bestPredictionListFromNumpy[i][0]+meanY)
            LINESTRING+=(pointStr)+","
        LINESTRING=LINESTRING[0:len(LINESTRING)-1]  # 去最后一个逗号
        LINESTRING="LINESTRING("+LINESTRING+")"
        print("陆地边界修正之后的拟合线为：（也就是有台阶的拟合线）")
        print(LINESTRING)


    # 最终要输出到数据库中的
    if(tag==0):
        # 最终输出到数据库的
        LINESTRING=""
        for i in range(len(afterIgnoredPredictionList)):
            pointStr=str(afterIgnoredPredictionList[i][0]+meanY)+" "+str(afterIgnoredXList[i][0]+meanX)
            LINESTRING+=(pointStr)+","
        LINESTRING=LINESTRING[0:len(LINESTRING)-1]  # 去最后一个逗号
        LINESTRING="LINESTRING("+LINESTRING+")"
        print("陆地边界修正之后并修复台阶后的拟合线为：")
        print(LINESTRING)
        return LINESTRING, bestLoss
    else:
        # 最终输出到数据库的
        LINESTRING=""
        for i in range(len(afterIgnoredPredictionList)):
            pointStr=str(afterIgnoredXList[i][0]+meanX)+" "+str(afterIgnoredPredictionList[i][0]+meanY)
            LINESTRING+=(pointStr)+","
        LINESTRING=LINESTRING[0:len(LINESTRING)-1]  # 去最后一个逗号
        LINESTRING="LINESTRING("+LINESTRING+")"
        print("陆地边界修正之后并修复台阶后的拟合线为：")
        print(LINESTRING)
        return LINESTRING, bestLoss



if __name__=='__main__':
    portA_lot = 140.7024
    portA_lat = 35.9531333
    portB_lot = 141.9666167
    portB_lat = 39.63415
    line1="LINESTRING(114.61057614063479 22.476878155853484,114.617015 22.4792816667,114.730146667 22.497405,114.84344 22.5240616667,114.905355 22.538305,114.9503 22.5505716667,115.061471667 22.571645,115.16411 22.5804933333,115.170391667 22.5807283333,115.282101667 22.58913,115.3957 22.59377,115.502356667 22.6038083333,115.670665 22.65297,115.958921667 22.7342416667,116.054403333 22.7595416667,116.150483333 22.788465,116.238551667 22.8159266667,116.25256 22.82103,116.35028 22.85064,116.357263333 22.8529166667,116.455143333 22.8849833333,116.529635 22.9327066667,116.530066667 22.9331533333,116.591955 22.988385,116.599821667 22.997015,116.60798378965885 23.004856646999222)"
    line2="LINESTRING(116.67952377637913 22.986123498774653,116.649301667 22.95034,116.635395 22.93577,116.503628333 22.8009166667,116.372718333 22.67765,116.343506667 22.66421,116.056443333 22.5463066667,115.847941667 22.490615,115.639151667 22.4351166667,115.440148333 22.3902166667,115.344721667 22.3759033333,115.242515 22.36558,115.049436667 22.3226066667,114.862676667 22.285675,114.665986667 22.280045,114.53023284653196 22.3272224655498)"
    line3="LINESTRING(116.69414745557131 22.98468319208877,116.678786667 22.96516,116.520633333 22.8101083333,116.379628333 22.6786683333,116.352818333 22.666405,116.142511667 22.5821533333,115.866816667 22.4933583333,115.645375 22.4372966667,115.422476667 22.3876666667,115.222563333 22.3562516667,115.007036667 22.3196966667,114.80591 22.2906066667,114.595016667 22.323295,114.53134268518525 22.328574807861667)"
    line4="LINESTRING(116.71743103238083 22.982389959933624,116.675641667 22.8427266667,116.572956667 22.7401916667,116.447648333 22.65444,116.242106667 22.5320633333,116.178366667 22.500475,115.93643 22.378275,115.9099 22.366645,115.706673333 22.28028,115.312025 22.2450683333,114.996841667 22.2537066667,114.755511667 22.259815,114.577348333 22.28323,114.53673853042469 22.335149664309448)"
    line5="LINESTRING(116.68357978506562 22.98572401678709,116.591536667 22.89948,116.046263333 22.5677116667,116.0444 22.5673533333,115.775671667 22.4713883333,115.717481667 22.46487,115.415491667 22.381705,114.56844300111669 22.37743311690889)"
    lineList = [line1,line2, line3, line4, line5]
    tup = linearRegress(portA_lot, portA_lat, portB_lot, portB_lat, lineList)
    # print(type(tup))
    # print(tup[0])
    # print(tup[1])

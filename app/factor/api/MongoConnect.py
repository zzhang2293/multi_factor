from pymongo import MongoClient

class MongoConnect():
    def __init__(self):
        self.Client = None
        self.NowDB = None   #库链接
        self.NowDBName = None   #库名称
        self.NowCollection = None   #集合链接
        self.NowCollectionName = None #集合名称

    def Connect(self,IpAddr,Port,User,Pwd,DBName):
        '''
        :param IpAddr: 192.168.1.18
        :param Port: 17926
        :param User: fww
        :param Pwd: fww8888
        :param DBName: admin
        :return:
        '''
        NowUrl = "mongodb://%s:%s@%s:%s/?authSource=%s"%(User,Pwd,IpAddr,Port,DBName)
        #NowUrl = "mongodb://zyhjctpk:zyhjctpk88@192.168.3.68:27017/?authSource=CTPK"
        self.Client = MongoClient(NowUrl)
        try:
            self.NowDB = self.Client[DBName]
            self.NowDBName = DBName
            return 1
        except:
            return 0

    def Close(self):
        if self.Client and self.NowDB:
            self.Client.close()
            
    def Insert_One(self,DBName,CollectionName,MapData):
        '''
        插入一条数据，数据格式为Dict类型。如：{xxx:xxx,xxx:xxx}
        '''
        if self.NowDBName != DBName:
            self.NowDB = self.Client[DBName]
            self.NowDBName = DBName
        if self.NowCollectionName != CollectionName:
            self.NowCollection = self.NowDB[CollectionName]
            self.NowCollectionName = CollectionName

        result = self.NowCollection.insert(MapData)

    def Insert_Many(self,DBName,CollectionName,MapDataList):
        '''
        插入多条数据，数据格式为Dict类型的List。如：[{xxx:xxx,xxx:xxx},{xxx:xxx,xxx:xxx}, ...]
        '''
        if self.NowDBName != DBName:
            self.NowDB = self.Client[DBName]
            self.NowDBName = DBName
        if self.NowCollectionName != CollectionName:
            self.NowCollection = self.NowDB[CollectionName]
            self.NowCollectionName = CollectionName
        try:
            result = self.NowCollection.insert_many(MapDataList)
            return result
        except Exception as e:
            print(e.args)
            return 0

    def Select(self,DBName,CollectionName,FindMapData,feildMap):
        '''
        查询符合条件的数据,FindMapData为筛选条件，格式为Dict类型，{xxx:xxx,xxx:xxx, ...}
        '''
        if self.NowDBName != DBName:
            self.NowDB = self.Client[DBName]
            self.NowDBName = DBName
        if self.NowCollectionName != CollectionName:
            self.NowCollection = self.NowDB[CollectionName]
            self.NowCollectionName = CollectionName

        results = self.NowCollection.find(FindMapData,feildMap)
        return results

    def SelectCount(self,DBName,CollectionName,FindMapData):
        '''
        查询符合条件的数据条数,FindMapData为筛选条件，格式为Dict类型，{xxx:xxx,xxx:xxx, ...}
        '''
        if self.NowDBName != DBName:
            self.NowDB = self.Client[DBName]
            self.NowDBName = DBName
        if self.NowCollectionName != CollectionName:
            self.NowCollection = self.NowDB[CollectionName]
            self.NowCollectionName = CollectionName

        results = self.NowCollection.find(FindMapData).count()
        return results

    def Drop(self,DBName,CollectionName,DropMapData):
        '''
        删除数据，DropMapData为筛选条件，格式为Dict类型，{xxx:xxx,xxx:xxx, ...}
        '''
        if self.NowDBName != DBName:
            self.NowDB = self.Client[DBName]
            self.NowDBName = DBName
        if self.NowCollectionName != CollectionName:
            self.NowCollection = self.NowDB[CollectionName]
            self.NowCollectionName = CollectionName

        results = self.NowCollection.remove(DropMapData)


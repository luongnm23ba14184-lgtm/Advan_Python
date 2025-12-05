class Student:
    def __init__(self,id,name,DoB):
        self.__id=id
        self.__name=name
        self.__DoB=DoB
        self.__mark=0.0

    def getid(self):
        return self.__id
    def getname(self):
        return self.__name
    def getdate(self):
        return self.__DoB
    def getmark(self):
        return self.__mark

    def setid(self,id):
        self.__id=id
    def setname(self,name):
        self.__name=name
    def setdate(self,DoB):
        self.__date=DoB
    def setmark(self,mark):
        self.__mark=mark
    def __str__(self):
        return "(" +self.__id +","+self._name+")" 
class Course:
    def __int__(self,name,id):
        self.__id=id
        self.__name=name
        self.__studentList=[]
    def getid(self):
        return self.__id
    def getname(self):
        return self.__name
    def setid(self,id):
        self.__id=id
    def setname(self,name):
        self.__name=name
    def addStudent(self,stu):
        self.__studentList.append(stu)
    def listingAllStudent(self):
        for stu in self.__studentList:
            print(stu)
    def updatamark(self,mark):
        for i, stu in enumerate(self.__studentList):
            stu.setmark(mark[i])
    def showmark(self):
        for stu in self.__studentList:
            print(f"{stu} have {stu.getmark()} for{self.__id}")
        print()
    
        
            



            





        
    





    


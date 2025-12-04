class Student:
    def __init__(self, id, name, DoB):
        self.__id = id
        self.__name = name
        self.__DoB = DoB
        self.__mark = 0.0
        
    def getId(self):
        return self.__id
        
    def getName(self):
        return self.__name
        
    def getDate(self):
        return self.__DoB
        
    def getMark(self):
        return self.__mark
        
    def setId(self, id):
        self.__id = id
        
    def setName(self, name):
        self.__name = name
        
    def setDate(self, DoB):
        self.__DoB = DoB
    
    def setMark(self, mark):
        self.__mark = mark
        
    def __str__(self):
        return "(" + self.__id + ", " + self.__name + ")"
        
class Course:
    def __init__(self, id, name):
        self.__id = id
        self.__name = name
        self.__stuList = []
        
    def getId(self):
        return self.__id
        
    def getName(self):
        return self.__name
        
    def setId(self, id):
        self.__id = id
        
    def setName(self, name):
        self.__name = name

    def addStu(self, stu):
        self.__stuList.append(stu)
        
    def listingAllStudent(self):
        for stu in self.__stuList:
            print(stu)
            
    def updateMark(self, marks):
        for i, stu in enumerate(self.__stuList):
            stu.setMark(marks[i])
            
    def showMark(self):
        for stu in self.__stuList:
            print(f"{stu} have {stu.getMark()} for {self.__id}")
            
        print()
            
    def haveStuId(self, id):
        if not any(stu.getId() == id for stu in self.__stuList):
            return 0
    
        return 1
        
    def __str__(self):
        return "- " + self.__name + " (" + self.__id + ")\n"
        
class SystemManageMark:
    def __init__(self):
        self.__courses = []
        
    def addNewCourse(self, course):
        self.__courses.append(course)
        
    def showCourseList(self):
        print(f"Course available :\n")
        for course in self.__courses:
            print(course)
    
    def haveCourseId(self, courseId):
        if not any(course.getId() == courseId for course in self.__courses):
            return 0
    
        return 1
            
    def updateMark4Course(self, courseId, marks):
        for course in self.__courses:
            if course.getId() == courseId:
                course.updateMark(marks)
                
                return "Update conplete!"
            
    def showMark4Course(self, courseId):
        for course in self.__courses:
            if course.getId() == courseId:
                course.showMark()
                
                return 0
            
# ------------- MAIN -------------

students = []
courses = [] 

def inputStudentList():
    n = int(input("Enter number of students : "))

    for i in range(n):
        id, name, DoB = input(f"Enter information of student {i + 1} : ").split()
        students.append(Student(id, name, DoB))    
    
        
def inputCourseList():
    n = int(input("Enter number of courses : "))

    for i in range(n):
        id, name= input(f"Enter information of course {i + 1} : ").split()
        course = Course(id, name)
        
        for stu in students:
            course.addStu(stu)
            
        courses.append(course)
        
def chooseCourse():
    system.showCourseList()
    
    courseId = input("Choose a course : ")
    
    while not system.haveCourseId(courseId):
        courseId = input("Choose other course : ")
        
    return courseId
    
def readMark4Course(courseId):
    marks = []
    for stu in students:
        marks.append(float(input(f"Mark of {stu} for {courseId} : ")))
        
    return marks

def builtSystem():
    for course in courses:
        system.addNewCourse(course)
           

inputStudentList()
inputCourseList()

system = SystemManageMark()

builtSystem()

courseId = chooseCourse()
marks = readMark4Course(courseId)
system.updateMark4Course(courseId, marks)

print()
system.showMark4Course(courseId)
students = []
courses = []
marks = []

def validStudentFormat(line):
    return len(line.split("|")) == 3

def validCourseFormat(line):
    return len(line.split("|")) == 2

def validMarkFormat(line):
    return len(line.split("|")) == 2


def inputStudent():
    line = input()
    
    while not validStudentFormat(line):
        line = input()
        
    return line.split("|")
    
def inputCourse():
    line = input()
    
    while not validCourseFormat(line):
        line = input()
        
    return line.split("|")

def inputMark():
    line = input()
    
    while not validMarkFormat(line):
        line = input()
        
    return line.split("|")

def inputValidMark():
    student_id, mark = inputMark()
    
    while not any(student_id == student["id"] for student in students):
        student_id, mark = inputMark()
        
    return student_id, mark


def inputStudentList():
    n = int(input())
    
    for i in range(n):
        id, name, DoB = inputStudent()
        students.append({
            "id" : id,
            "name" : name,
            "DoB" : DoB
        })
        
def inputCourseList():
    n = int(input())
    
    for i in range(n):
        id, name = inputCourse()
        courses.append({
            "id" : id,
            "name" : name,
        })    
    
def inputMarkList():
    print("Valid Course : \n")
    for course in courses:
        print(f'{course["id"]} - {course["name"]}\n')
        
    course_id = input("Choose a course : ")
    
    while not any(course_id == course["id"] for course in courses):
        print("Choose other course!\n")
        course_id = input("Choose a course : ")
       
    n = int(input())
    
    for i in range(n):
        student_id, mark = inputValidMark()
        marks.append({
            "course_id" : course_id,
            "student_id" : student_id,
            "mark" : mark
        })
        
def listingStudent():
    print(students)
    
def listingCourse():
    print(courses)
    
def listingMark():
    print(marks)
    
inputStudentList()
inputCourseList()
inputMarkList()

listingCourse()
listingMark()
listingStudent()
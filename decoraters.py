# def div(a,b):
#     print(a/b)
#
# def smart_div(func):
#     def inner(a,b):
#         if a<b:
#             a,b=b,a
#         return func(a,b)
#     return inner
#
# div = smart_div(div)
# div(2,4)
# def f1():
#     print('called f1')
#
# def f2(f):
#     f()
# def f1(func):
#     def wrapper(*args,**kwargs):
#         print('started')
#         val = func(*args,**kwargs)
#         print('ended')
#         return val
#
#     return wrapper
# @f1
# def f(a,b=0):
#     print(a,b)
#
# @f1
# def add(x,y):
#     return x+y
#
# print(f(4,5))
import time

def timer(func):
    def wrapper():
        before = time.time()
        func()
        print('functiom took:', time.time()-before,'seconds')

    return wrapper

@timer
def run():
    time.sleep(5)

run()
import time, os, sys
from fastapi import HTTPException
import traceback
from functools import wraps

def exceptionhandler(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            error_stack = traceback.format_exception(exc_type, exc_obj, exc_tb)
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            error_stack = error_stack[1:-1]
            error_stack = [i.strip() for i in error_stack]
            lib_lst = []
            code_lst = []
            for x in error_stack:
                f = x.startswith('File')
                if f == True:
                    y = x.split(",")[0]
                    path = y.replace('File', '').strip()
                    path = path[1:-1]
                    z = x.split(",")[1]
                    line = int(z.replace('line','').strip())
                    f = x.split(",")[2]
                    function = f.replace('in','').strip()
                    function = function.split(" ")[0]
                    function = function.replace('/n','').strip()
                    lib_path ={"path":path,"line":line, "function":function}
                    if "lib" in path:
                        lib_lst.append(lib_path)
                    else:
                        code_lst.append(lib_path)
            new_dct = {"site-packages": lib_lst,"code":code_lst}
            obj = {"error_type":str(exc_type), "error_desc":str(exc_obj), "traceback":new_dct}
            raise HTTPException(status_code=422, detail=obj)

    return wrapper
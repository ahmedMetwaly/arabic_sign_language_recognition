"""
Gunicorn configuration file.
"""

import multiprocessing


#
# Worker processes
#

workers = 1
worker_class = "eventlet"
timeout = 60
worker_tmp_dir = "/dev/shm"
logfile = "-"

#
#   Logging
#

errorlog = "-"
# accesslog = "-"
loglevel = "info"


# h 	remote address
# l 	'-'
# u 	user name
# t 	date of the request
# r 	status line (e.g. GET / HTTP/1.1)
# m 	request method
# U 	URL path without query string
# q 	query string
# H 	protocol
# s 	status
# B 	response length
# b 	response length or '-' (CLF format)
# f 	referer
# a 	user agent
# T 	request time in seconds
# D 	request time in microseconds
# L 	request time in decimal seconds
# p 	process ID
# {header}i 	request header
# {header}o 	response header
# {variable}e 	environment variable
access_log_format = '%(t)s - %(s)s - "%(r)s" %(b)s '

#
# Server hooks
#


def pre_fork(server, worker):
    pass


def post_fork(server, worker):
    server.log.info(f"Worker spawned (pid: {worker.pid})")


def pre_exec(server):
    server.log.info("Forked child, re-executing.")


def when_ready(server):
    server.log.info("Server is ready. Spawning workers")
    open('/tmp/app-initialized', 'w').close()


def worker_int(worker):
    worker.log.info("worker received INT or QUIT signal")

    # get traceback info
    import threading
    import sys
    import traceback

    id2name = {th.ident: th.name for th in threading.enumerate()}
    code = []
    for threadId, stack in sys._current_frames().items():
        code.append(f"\n# Thread: {id2name.get(threadId, '')}({threadId})")
        for filename, lineno, name, line in traceback.extract_stack(stack):
            code.append(f'File: "{filename}", line {lineno}, in {name}')
            if line:
                code.append(f"  {line.strip()}")
    worker.log.debug("\n".join(code))


def worker_abort(worker):
    worker.log.info("worker received SIGABRT signal")

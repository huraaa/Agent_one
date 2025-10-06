# infra/retry.py
import time, random

def retry(fn, tries=3, base=0.5, max_delay=4.0, retry_on=(Exception,)):
    for i in range(tries):
        try:
            return fn()
        except retry_on as e:
            if i == tries-1: raise
            time.sleep(min(max_delay, base * (2**i)) + random.random()*0.1)

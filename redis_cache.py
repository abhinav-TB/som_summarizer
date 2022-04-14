import redis
import json
client = redis.Redis(host='redis-19834.c212.ap-south-1-1.ec2.cloud.redislabs.com', port=19834 ,password="1Ipa2vL4UlMUwSSt95Qpy1YZ4OZioPC2")


def check_cache(len,text):
  
  try:
    reply = json.loads(client.execute_command('JSON.GET', text))
    if reply['len'] == len:
      return reply['summary']
    else:
      print("cache miss")
      return None
  except:
    print("cache miss")
    return None


def add_item(len,text,summary):
  d = {
    'len' : len,
    'summary':summary
  }
  try:
    client.execute_command('JSON.SET', text, '.', json.dumps(d))
  except:
    print("insufficient memory")

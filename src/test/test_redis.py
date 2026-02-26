import redis

REDIS_PASSWORD = "RedisAuth"
REDIS_HOST = "localhost"
REDIS_PORT = 6446

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, username="default",  password=REDIS_PASSWORD, db=0)

# redisClient = redis.StrictRedis(host=REDIS_HOST,
#                                             port=REDIS_PORT,
#                                             password=REDIS_PASSWORD,
#                                             db=0)

try:
    r.ping()
    print("✅ Connected to Redis successfully!")
except redis.exceptions.ConnectionError as e:
    print("❌ Redis connection failed:", e)
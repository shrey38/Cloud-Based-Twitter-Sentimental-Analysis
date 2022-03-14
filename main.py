#Author   SHREEYASH PANDEY   TWITTER ANALYSIS



from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import Stream
import json
import boto3
from datetime import date
import uuid

# This is for the Authentication Purposes(consumer key, consumer secret, access token, access secret).

consumerkey = 'bua0yO8fyzxuhDYJbPUXOMm5c'
consumersecret = 'gj0qIfLfVNtS01rvEmQ3vnJ22ApdDSSA2LO5JI2groiVP3b0yD'
authitoken = '1492723939881414657-PZb2SNwzOblorF6fbZrM5YJZJVVG1X'
authisecret = 'RHIaVCbJ7KV6iDICLy5fxAHXNyWQZZQSmNCPkom71c2qx'

s3= boto3.client('s3')


class listener(Stream):

    def on_data(self, data):
        t_data = json.loads(data)

        tweet = t_data["text"]

        us_name = t_data["user"]["screen_name"]
        print(us_name)

        values = {"name": us_name, "id": t_data['id'], "data": tweet}

        day = date.today()
        day = day.strftime("%Y/%m/%d")
        u_id = uuid.uuid1()
        key_name = day + '/' + str(u_id) + '_' + us_name + '.json'
        s3.put_object(Body=str(values), Bucket='test-shrey-opinion123', Key=key_name)

        return True

    def on_error(self, status):
        print(status)


auth = OAuthHandler(consumerkey, consumersecret)
auth.set_access_token(authitoken, authisecret)

twitterStream = Stream(auth, listener())
tags = ['#Movie']
twitterStream.filter(track=tags)
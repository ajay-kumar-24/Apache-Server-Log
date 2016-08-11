# Databricks notebook source exported at Thu, 21 Jul 2016 04:42:17 UTC
# **Web Server Log Analysis with Apache Spark**
# Server log analysis is an ideal use case for Spark.  It's a very large, common data source and contains a rich set of information.  Spark allows you to store your logs in files on disk cheaply, while still providing a quick and simple way to perform data analysis on them.  This homework will show you how to use Apache Spark on real-world text-based production logs and fully harness the power of that data.  Log data comes from many sources, such as web, file, and compute servers, application logs, user-generated content,  and can be used for monitoring servers, improving business and customer intelligence, building recommendation systems, fraud detection, and much more.

labVersion = 'cs105x-lab2-1.1.0'
# Part 1: Introduction and Imports

throwaway_df = sqlContext.createDataFrame([('Anthony', 10), ('Julia', 20), ('Fred', 5)], ('name', 'count'))

import re
import datetime
from databricks_test_helper import Test

dir(sqlContext)


# Part 2: Exploratory Data Analysis
# Let's begin looking at our data.  For this lab, we will use a data set from NASA Kennedy Space Center web server in Florida. The full data set is freely available at <http://ita.ee.lbl.gov/html/contrib/NASA-HTTP.html>, and it contains all HTTP requests for two months. We are using a subset that only contains several days' worth of requests.  The log file has already been downloaded for you.

import sys
import os

log_file_path = 'dbfs:/' + os.path.join('databricks-datasets', 'cs100', 'lab2', 'data-001', 'apache.access.log.PROJECT')

# Loading the log file

base_df = sqlContext.read.text(log_file_path)
base_df.printSchema()

base_df.show(truncate=False)
# (2b) Parsing the log file

# common-logfile-format
#  _remotehost rfc931 authuser [date] "request" status bytes_

from pyspark.sql.functions import split, regexp_extract
split_df = base_df.select(regexp_extract('value', r'^([^\s]+\s)', 1).alias('host'),
                          regexp_extract('value', r'^.*\[(\d\d/\w{3}/\d{4}:\d{2}:\d{2}:\d{2} -\d{4})]', 1).alias('timestamp'),
                          regexp_extract('value', r'^.*"\w+\s+([^\s]+)\s+HTTP.*"', 1).alias('path'),
                          regexp_extract('value', r'^.*"\s+([^\s]+)', 1).cast('integer').alias('status'),
                          regexp_extract('value', r'^.*\s+(\d+)$', 1).cast('integer').alias('content_size'))
#split_df.show(truncate=False)

# (2c) Data Cleaning
# Verify that there are no null rows in the original data set.

base_df.filter(base_df['value'].isNull()).count()
bad_rows_df = split_df.filter(split_df['host'].isNull() |
                              split_df['timestamp'].isNull() |
                              split_df['path'].isNull() |
                              split_df['status'].isNull() |
                             split_df['content_size'].isNull())
bad_rows_df.count()

from pyspark.sql.functions import col, sum

def count_null(col_name):
  return sum(col(col_name).isNull().cast('integer')).alias(col_name)
exprs = []
for col_name in split_df.columns:
  exprs.append(count_null(col_name))


split_df.agg(*exprs).show()


bad_content_size_df = base_df.filter(~ base_df['value'].rlike(r'\d+$'))
bad_content_size_df.count()



from pyspark.sql.functions import lit, concat
# (2d) Fix the rows with null content\_size
cleaned_df = split_df.na.fill({'content_size': 0})

exprs = []
for col_name in cleaned_df.columns:
  exprs.append(count_null(col_name))

# (2e) Parsing the timestamp.
month_map = {
  'Jan': 1, 'Feb': 2, 'Mar':3, 'Apr':4, 'May':5, 'Jun':6, 'Jul':7,
  'Aug':8,  'Sep': 9, 'Oct':10, 'Nov': 11, 'Dec': 12
}

def parse_clf_time(s):
    return "{0:04d}-{1:02d}-{2:02d} {3:02d}:{4:02d}:{5:02d}".format(
      int(s[7:11]),
      month_map[s[3:6]],
      int(s[0:2]),
      int(s[12:14]),
      int(s[15:17]),
      int(s[18:20])
    )

u_parse_time = udf(parse_clf_time)

logs_df = cleaned_df.select('*', u_parse_time(cleaned_df['timestamp']).cast('timestamp').alias('time')).drop('timestamp')
total_log_entries = logs_df.count()
logs_df.printSchema()
display(logs_df)
logs_df.cache()

# Part 3: Analysis Walk-Through on the Web Server Log File
# Calculate statistics based on the content size.
content_size_summary_df = logs_df.describe(['content_size'])

from pyspark.sql import functions as sqlFunctions
content_size_stats =  (logs_df
                       .agg(sqlFunctions.min(logs_df['content_size']),
                            sqlFunctions.avg(logs_df['content_size']),
                            sqlFunctions.max(logs_df['content_size']))
                       .first())

print 'Using SQL functions:'
print 'Content Size Avg: {1:,.2f}; Min: {0:.2f}; Max: {2:,.0f}'.format(*content_size_stats)

# (3b) Example: HTTP Status Analysis

status_to_count_df =(logs_df
                     .groupBy('status')
                     .count()
                     .sort('status')
                     .cache())

status_to_count_length = status_to_count_df.count()
print 'Found %d response codes' % status_to_count_length

assert status_to_count_length == 7
assert status_to_count_df.take(100) == [(200, 940847), (302, 16244), (304, 79824), (403, 58), (404, 6185), (500, 2), (501, 17)]

# (3c) Example: Status Graphing


display(status_to_count_df)


log_status_to_count_df = status_to_count_df.withColumn('log(count)', sqlFunctions.log(status_to_count_df['count']))

display(log_status_to_count_df)

from spark_notebook_helpers import prepareSubplot, np, plt, cm

data = log_status_to_count_df.drop('count').collect()
x, y = zip(*data)
index = np.arange(len(x))
bar_width = 0.7
colorMap = 'Set1'
cmap = cm.get_cmap(colorMap)

fig, ax = prepareSubplot(np.arange(0, 6, 1), np.arange(0, 14, 2))
plt.bar(index, y, width=bar_width, color=cmap(0))
plt.xticks(index + bar_width/2.0, x)
display(fig)

# (3d) Example: Frequent Hosts
# Any hosts that has accessed the server more than 10 times.
host_sum_df =(logs_df
              .groupBy('host')
              .count())

host_more_than_10_df = (host_sum_df
                        .filter(host_sum_df['count'] > 10)
                        .select(host_sum_df['host']))

print 'Any 20 hosts that have accessed more then 10 times:\n'
host_more_than_10_df.show(truncate=False)

# (3e) Example: Visualizing Paths

paths_df = (logs_df
            .groupBy('path')
            .count()
            .sort('count', ascending=False))

paths_counts = (paths_df
                .select('path', 'count')
                .map(lambda r: (r[0], r[1]))
                .collect())

paths, counts = zip(*paths_counts)

colorMap = 'Accent'
cmap = cm.get_cmap(colorMap)
index = np.arange(1000)

fig, ax = prepareSubplot(np.arange(0, 1000, 100), np.arange(0, 70000, 10000))
plt.xlabel('Paths')
plt.ylabel('Number of Hits')
plt.plot(index, counts[:1000], color=cmap(0), linewidth=3)
plt.axhline(linewidth=2, color='#999999')
display(fig)

display(paths_df)

# (3f) Top Paths
# Top Paths
print 'Top Ten Paths:'

expected = [
  (u'/images/NASA-logosmall.gif', 59666),
  (u'/images/KSC-logosmall.gif', 50420),
  (u'/images/MOSAIC-logosmall.gif', 43831),
  (u'/images/USA-logosmall.gif', 43604),
  (u'/images/WORLD-logosmall.gif', 43217),
  (u'/images/ksclogo-medium.gif', 41267),
  (u'/ksc.html', 28536),
  (u'/history/apollo/images/apollo-logo1.gif', 26766),
  (u'/images/launch-logo.gif', 24742),
  (u'/', 20173)
]
assert paths_df.take(10) == expected, 'incorrect Top Ten Paths'

# Part 4: Analyzing Web Server Log File

from pyspark.sql.functions import desc
not200DF = logs_df.filter(logs_df['status'] != 200).groupBy('path').count().sort('count',ascending = False).select('path','count')
not200DF.show(10)
# Sorted DataFrame containing all paths and the number of times they were accessed with non-200 return code
logs_sum_df = not200DF.sort('count',ascending = False).select('path','count')
print 'Top Ten failed URLs:'
logs_sum_df.show(10, False)

# TEST Top ten error paths (4a)
top_10_err_urls = [(row[0], row[1]) for row in logs_sum_df.take(10)]
top_10_err_expected = [
  (u'/images/NASA-logosmall.gif', 8761),
  (u'/images/KSC-logosmall.gif', 7236),
  (u'/images/MOSAIC-logosmall.gif', 5197),
  (u'/images/USA-logosmall.gif', 5157),
  (u'/images/WORLD-logosmall.gif', 5020),
  (u'/images/ksclogo-medium.gif', 4728),
  (u'/history/apollo/images/apollo-logo1.gif', 2907),
  (u'/images/launch-logo.gif', 2811),
  (u'/', 2199),
  (u'/images/ksclogosmall.gif', 1622)
]
Test.assertEquals(logs_sum_df.count(), 7675, 'incorrect count for logs_sum_df')
Test.assertEquals(top_10_err_urls, top_10_err_expected, 'incorrect Top Ten failed URLs')

# (4b) Exercise: Number of Unique Hosts
unique_host_count = logs_df.select('host').distinct().count()
unique_host_count.show()
print 'Unique hosts: {0}'.format(unique_host_count)

Test.assertEquals(unique_host_count, 54507, 'incorrect unique_host_count')
# (4c) Exercise: Number of Unique Daily Hosts
from pyspark.sql.functions import dayofmonth

day_to_host_pair_df = logs_df.select('host',dayofmonth('time').alias('day'))
day_group_hosts_df = day_to_host_pair_df.distinct()
daily_hosts_df = day_group_hosts_df.groupBy('day').count()
daily_hosts_df.cache()
print 'Unique hosts per day:'
daily_hosts_df.show(30, False)

daily_hosts_list = (daily_hosts_df
                    .map(lambda r: (r[0], r[1]))
                    .take(30))

Test.assertEquals(day_to_host_pair_df.count(), total_log_entries, 'incorrect row count for day_to_host_pair_df')
Test.assertEquals(daily_hosts_df.count(), 21, 'incorrect daily_hosts_df.count()')
Test.assertEquals(daily_hosts_list, [(1, 2582), (3, 3222), (4, 4190), (5, 2502), (6, 2537), (7, 4106), (8, 4406), (9, 4317), (10, 4523), (11, 4346), (12, 2864), (13, 2650), (14, 4454), (15, 4214), (16, 4340), (17, 4385), (18, 4168), (19, 2550), (20, 2560), (21, 4134), (22, 4456)], 'incorrect daily_hosts_df')
Test.assertTrue(daily_hosts_df.is_cached, 'incorrect daily_hosts_df.is_cached')

# (4d) Exercise: Visualizing the Number of Unique Daily Hosts

days_with_hosts = []#daily_hosts_df.select(collect_list('day'))
hosts = []
for day in daily_hosts_df.collect():
  days_with_hosts.append(day[0])
  hosts.append(day[1])
print(days_with_hosts)
print(hosts)

# TEST Visualizing unique daily hosts (4d)
test_days = range(1, 23)
test_days.remove(2)
Test.assertEquals(days_with_hosts, test_days, 'incorrect days')
Test.assertEquals(hosts, [2582, 3222, 4190, 2502, 2537, 4106, 4406, 4317, 4523, 4346, 2864, 2650, 4454, 4214, 4340, 4385, 4168, 2550, 2560, 4134, 4456], 'incorrect hosts')
fig, ax = prepareSubplot(np.arange(0, 30, 5), np.arange(0, 5000, 1000))
colorMap = 'Dark2'
cmap = cm.get_cmap(colorMap)
plt.plot(days_with_hosts, hosts, color=cmap(0), linewidth=3)
plt.axis([0, max(days_with_hosts), 0, max(hosts)+500])
plt.xlabel('Day')
plt.ylabel('Hosts')
plt.axhline(linewidth=3, color='#999999')
plt.axvline(linewidth=2, color='#999999')
display(fig)


display(daily_hosts_df)

# (4e) Exercise: Average Number of Daily Requests per Host

total_req_per_day_df = logs_df.select(dayofmonth('time').alias('day')).groupBy('day').count()
print daily_hosts_df[0]
avg_daily_req_per_host_df = (
total_req_per_day_df.join(daily_hosts_df,'day').select(total_req_per_day_df.day, (total_req_per_day_df['count']/ daily_hosts_df['count']).alias('avg_reqs_per_host_per_day')))

print 'Average number of daily requests per Hosts is:\n'
avg_daily_req_per_host_df.cache()

# TEST Average number of daily requests per hosts (4e)
avg_daily_req_per_host_list = (
  avg_daily_req_per_host_df.select('day', avg_daily_req_per_host_df['avg_reqs_per_host_per_day'].cast('integer').alias('avg_requests'))
                           .collect()
)

values = [(row[0], row[1]) for row in avg_daily_req_per_host_list]
print values
Test.assertEquals(values, [(1, 13), (3, 12), (4, 14), (5, 12), (6, 12), (7, 13), (8, 13), (9, 14), (10, 13), (11, 14), (12, 13), (13, 13), (14, 13), (15, 13), (16, 13), (17, 13), (18, 13), (19, 12), (20, 12), (21, 13), (22, 12)], 'incorrect avgDailyReqPerHostDF')
Test.assertTrue(avg_daily_req_per_host_df.is_cached, 'incorrect avg_daily_req_per_host_df.is_cached')

# (4f) Exercise: Visualizing the Average Daily Requests per Unique Host

days_with_avg = avg_daily_req_per_host_df.select('day').collect()
avgs = avg_daily_req_per_host_df.select('avg_reqs_per_host_per_day').collect()
for day in range(0,len(days_with_avg)):
  days_with_avg[day] = days_with_avg[day][0]
  avgs[day] = avgs[day][0]
  
  

print(days_with_avg)
print(avgs)

# TEST Average Daily Requests per Unique Host (4f)
Test.assertEquals(days_with_avg, [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22], 'incorrect days')
Test.assertEquals([int(a) for a in avgs], [13, 12, 14, 12, 12, 13, 13, 14, 13, 14, 13, 13, 13, 13, 13, 13, 13, 12, 12, 13, 12], 'incorrect avgs')

fig, ax = prepareSubplot(np.arange(0, 20, 5), np.arange(0, 16, 2))
colorMap = 'Set3'
cmap = cm.get_cmap(colorMap)
plt.plot(days_with_avg, avgs, color=cmap(0), linewidth=3)
plt.axis([0, max(days_with_avg), 0, max(avgs)+2])
plt.xlabel('Day')
plt.ylabel('Average')
plt.axhline(linewidth=3, color='#999999')
plt.axvline(linewidth=2, color='#999999')
display(fig)

display(fig)

# Part 5: Exploring 404 Status Codes
# (5a) Exercise: Counting 404 Response Codes

not_found_df = logs_df.filter('status = 404')
not_found_df.cache()
print('Found {0} 404 URLs').format(not_found_df.count())
Test.assertEquals(not_found_df.count(), 6185, 'incorrect not_found_df.count()')
Test.assertTrue(not_found_df.is_cached, 'incorrect not_found_df.is_cached')

# (5b) Exercise: Listing 404 Status Code Records

not_found_paths_df = not_found_df.select('path')
unique_not_found_paths_df = not_found_paths_df.distinct()

print '404 URLS:\n'
# TEST Listing 404 records (5b)

bad_unique_paths_40 = set([row[0] for row in unique_not_found_paths_df.take(40)])
Test.assertEquals(len(bad_unique_paths_40), 40, 'bad_unique_paths_40 not distinct')

# (5c) Exercise: Listing the Top Twenty 404 Response Code paths

top_20_not_found_df = not_found_paths_df.groupBy('path').count().sort('count', ascending = False)

print 'Top Twenty 404 URLs:\n'
top_20_not_found_df.show(n=20, truncate=False)

top_20_not_found = [(row[0], row[1]) for row in top_20_not_found_df.take(20)]
top_20_expected = [
 (u'/pub/winvn/readme.txt', 633),
 (u'/pub/winvn/release.txt', 494),
 (u'/shuttle/missions/STS-69/mission-STS-69.html', 430),
 (u'/images/nasa-logo.gif', 319),
 (u'/elv/DELTA/uncons.htm', 178),
 (u'/shuttle/missions/sts-68/ksc-upclose.gif', 154),
 (u'/history/apollo/sa-1/sa-1-patch-small.gif', 146),
 (u'/images/crawlerway-logo.gif', 120),
 (u'/://spacelink.msfc.nasa.gov', 117),
 (u'/history/apollo/pad-abort-test-1/pad-abort-test-1-patch-small.gif', 100),
 (u'/history/apollo/a-001/a-001-patch-small.gif', 97),
 (u'/images/Nasa-logo.gif', 85),
 (u'', 76),
 (u'/shuttle/resources/orbiters/atlantis.gif', 63),
 (u'/history/apollo/images/little-joe.jpg', 62),
 (u'/images/lf-logo.gif', 59),
 (u'/shuttle/resources/orbiters/discovery.gif', 56),
 (u'/shuttle/resources/orbiters/challenger.gif', 54),
 (u'/robots.txt', 53),
 (u'/history/apollo/pad-abort-test-2/pad-abort-test-2-patch-small.gif', 38)
]
Test.assertEquals(top_20_not_found, top_20_expected, 'incorrect top_20_not_found')

# (5d) Exercise: Listing the Top Twenty-five 404 Response Code Hosts

hosts_404_count_df = not_found_df.groupBy('host').count().sort(desc('count'))

print 'Top 25 hosts that generated errors:\n'
hosts_404_count_df.show(n=25, truncate=False)


top_25_404 = [(row[0], row[1]) for row in hosts_404_count_df.take(25)]
Test.assertEquals(len(top_25_404), 25, 'length of errHostsTop25 is not 25')

expected = set([
  (u'maz3.maz.net ', 39),
  (u'piweba3y.prodigy.com ', 39),
  (u'gate.barr.com ', 38),
  (u'nexus.mlckew.edu.au ', 37),
  (u'ts8-1.westwood.ts.ucla.edu ', 37),
  (u'm38-370-9.mit.edu ', 37),
  (u'204.62.245.32 ', 33),
  (u'spica.sci.isas.ac.jp ', 27),
  (u'163.206.104.34 ', 27),
  (u'www-d4.proxy.aol.com ', 26),
  (u'203.13.168.17 ', 25),
  (u'203.13.168.24 ', 25),
  (u'www-c4.proxy.aol.com ', 25),
  (u'internet-gw.watson.ibm.com ', 24),
  (u'crl5.crl.com ', 23),
  (u'piweba5y.prodigy.com ', 23),
  (u'scooter.pa-x.dec.com ', 23),
  (u'onramp2-9.onr.com ', 22),
  (u'slip145-189.ut.nl.ibm.net ', 22),
  (u'198.40.25.102.sap2.artic.edu ', 21),
  (u'msp1-16.nas.mr.net ', 20),
  (u'gn2.getnet.com ', 20),
  (u'tigger.nashscene.com ', 19),
  (u'dial055.mbnet.mb.ca ', 19),
  (u'isou24.vilspa.esa.es ', 19)
])
Test.assertEquals(len(set(top_25_404) - expected), 0, 'incorrect hosts_404_count_df')

# (5e) Exercise: Listing 404 Errors per Day
errors_by_date_sorted_df = not_found_df.select(dayofmonth('time').alias('day')).groupBy('day').count()
errors_by_date_sorted_df.cache()
print '404 Errors by day:\n'
errors_by_date_sorted_df.show()


errors_by_date = [(row[0], row[1]) for row in errors_by_date_sorted_df.collect()]
expected = [
  (1, 243),
  (3, 303),
  (4, 346),
  (5, 234),
  (6, 372),
  (7, 532),
  (8, 381),
  (9, 279),
  (10, 314),
  (11, 263),
  (12, 195),
  (13, 216),
  (14, 287),
  (15, 326),
  (16, 258),
  (17, 269),
  (18, 255),
  (19, 207),
  (20, 312),
  (21, 305),
  (22, 288)
]
Test.assertEquals(errors_by_date, expected, 'incorrect errors_by_date_sorted_df')
Test.assertTrue(errors_by_date_sorted_df.is_cached, 'incorrect errors_by_date_sorted_df.is_cached')

# (5f) Exercise: Visualizing the 404 Errors by Day

days_with_errors_404 = errors_by_date_sorted_df.collect()
errors_404_by_day = errors_by_date_sorted_df.collect()
for day in range(0,len(days_with_errors_404)):
  days_with_errors_404[day] = days_with_errors_404[day][0]
  errors_404_by_day[day] = errors_404_by_day[day][1]

print days_with_errors_404
print errors_404_by_day

# TEST Visualizing the 404 Response Codes by Day (4f)
Test.assertEquals(days_with_errors_404, [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22], 'incorrect days_with_errors_404')
Test.assertEquals(errors_404_by_day, [243, 303, 346, 234, 372, 532, 381, 279, 314, 263, 195, 216, 287, 326, 258, 269, 255, 207, 312, 305, 288], 'incorrect errors_404_by_day')

fig, ax = prepareSubplot(np.arange(0, 20, 5), np.arange(0, 600, 100))
colorMap = 'rainbow'
cmap = cm.get_cmap(colorMap)
plt.plot(days_with_errors_404, errors_404_by_day, color=cmap(0), linewidth=3)
plt.axis([0, max(days_with_errors_404), 0, max(errors_404_by_day)])
plt.xlabel('Day')
plt.ylabel('404 Errors')
plt.axhline(linewidth=3, color='#999999')
plt.axvline(linewidth=2, color='#999999')
display(fig)

display(errors_by_date_sorted_df)

# (5g) Exercise: Top Five Days for 404 Errors

top_err_date_df = errors_by_date_sorted_df.sort(desc('count'))

print 'Top Five Dates for 404 Requests:\n'
top_err_date_df.show(5)

Test.assertEquals([(r[0], r[1]) for r in top_err_date_df.take(5)], [(7, 532), (8, 381), (6, 372), (4, 346), (15, 326)], 'incorrect top_err_date_df')

# (5h) Exercise: Hourly 404 Errors
from pyspark.sql.functions import hour
hour_records_sorted_df = not_found_df.select(hour('time').alias('hour')).groupBy('hour').count()

print 'Top hours for 404 requests:\n'
hour_records_sorted_df.cache()
hour_records_sorted_df.show(24)

# TEST Hourly 404 response codes (5h)

errs_by_hour = [(row[0], row[1]) for row in hour_records_sorted_df.collect()]

expected = [
  (0, 175),
  (1, 171),
  (2, 422),
  (3, 272),
  (4, 102),
  (5, 95),
  (6, 93),
  (7, 122),
  (8, 199),
  (9, 185),
  (10, 329),
  (11, 263),
  (12, 438),
  (13, 397),
  (14, 318),
  (15, 347),
  (16, 373),
  (17, 330),
  (18, 268),
  (19, 269),
  (20, 270),
  (21, 241),
  (22, 234),
  (23, 272)
]
Test.assertEquals(errs_by_hour, expected, 'incorrect errs_by_hour')
Test.assertTrue(hour_records_sorted_df.is_cached, 'incorrect hour_records_sorted_df.is_cached')

# (5i) Exercise: Visualizing the 404 Response Codes by Hour

hours_with_not_found = hour_records_sorted_df.collect()
not_found_counts_per_hour = hour_records_sorted_df.collect()
for day in range(0,len(hours_with_not_found)):
  hours_with_not_found[day] = hours_with_not_found[day][0]
  not_found_counts_per_hour[day] = not_found_counts_per_hour[day][1]

print hours_with_not_found
print not_found_counts_per_hour

# TEST Visualizing the 404 Response Codes by Hour (5i)
Test.assertEquals(hours_with_not_found, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23], 'incorrect hours_with_not_found')
Test.assertEquals(not_found_counts_per_hour, [175, 171, 422, 272, 102, 95, 93, 122, 199, 185, 329, 263, 438, 397, 318, 347, 373, 330, 268, 269, 270, 241, 234, 272], 'incorrect not_found_counts_per_hour')

fig, ax = prepareSubplot(np.arange(0, 25, 5), np.arange(0, 500, 50))
colorMap = 'seismic'
cmap = cm.get_cmap(colorMap)
plt.plot(hours_with_not_found, not_found_counts_per_hour, color=cmap(0), linewidth=3)
plt.axis([0, max(hours_with_not_found), 0, max(not_found_counts_per_hour)])
plt.xlabel('Hour')
plt.ylabel('404 Errors')
plt.axhline(linewidth=3, color='#999999')
plt.axvline(linewidth=2, color='#999999')
display(fig)

display(hour_records_sorted_df)

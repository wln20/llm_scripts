"""
Directly use the tavily search engine tool's API and return search results.
Reference: https://python.langchain.com/docs/tutorials/agents/
Dependencies:
```
pip install langchain-community
pip install tavily-python
```
"""
import os
from langchain_community.tools.tavily_search import TavilySearchResults

os.environ["TAVILY_API_KEY"] = # TODO

# Creat tool (this time it's a Tavily search engine)
search = TavilySearchResults(max_results=2)
# Directly use the search engine tool
search_msg_1 = "what is the weather in SF"
search_results = search.invoke(search_msg_1)
print(len(search_results))  # 2 (max_results)
print(search_results[0])
"""
{
    'url': 'https://www.weatherapi.com/', 
    'content': "{'location': {'name': 'San Francisco', 'region': 'California', 'country': 'United States of America', 'lat': 37.775, 'lon': -122.4183, 'tz_id': 'America/Los_Angeles', 'localtime_epoch': 1735534372, 'localtime': '2024-12-29 20:52'}, 'current': {'last_updated_epoch': 1735533900, 'last_updated': '2024-12-29 20:45', 'temp_c': 12.2, 'temp_f': 54.0, 'is_day': 0, 'condition': {'text': 'Partly cloudy', 'icon': '//cdn.weatherapi.com/weather/64x64/night/116.png', 'code': 1003}, 'wind_mph': 7.8, 'wind_kph': 12.6, 'wind_degree': 283, 'wind_dir': 'WNW', 'pressure_mb': 1024.0, 'pressure_in': 30.25, 'precip_mm': 0.0, 'precip_in': 0.0, 'humidity': 80, 'cloud': 25, 'feelslike_c': 10.9, 'feelslike_f': 51.6, 'windchill_c': 9.7, 'windchill_f': 49.4, 'heatindex_c': 11.1, 'heatindex_f': 51.9, 'dewpoint_c': 9.4, 'dewpoint_f': 48.8, 'vis_km': 16.0, 'vis_miles': 9.0, 'uv': 0.0, 'gust_mph': 11.8, 'gust_kph': 19.0}}"
}
"""
print(search_results[1])
"""
{
    'url': 'https://www.peoplesweather.com/weather/San+Francisco/?date=2024-12-30', 
    'content': 'Weather for San Francisco | People°s Weather Home Weather News MyPhoto Competitions Contact Us Home Weather Forecast News & Highlights MyPhoto Competitions Contact Us Get the weather for Johannesburg Weather for San Francisco Weather United States San Francisco Monday 30 December 2024 8°CFeels like: 8°CNW / 7km/hLight BreezePartly Cloudy. Cool Pressure1024mbHumidity86%Rain0%Cloud Cover27%Dew Point6°C This Afternoon12°CN / 7km/hLight BreezeOvercast. Cool Tonight10°CWNW / 11km/hLight BreezePartly Cloudy. Cool 6 Day Forecast for San Francisco Weather Detailed Forecast SA National Parks iSimangaliso Popular Submit your Photo Contact us Careers News Room Newsletter Subscribe Now Terms of Use Privacy Sitemap © 2007-2024 People°s Weather Pty. Ltd., All rights reserved.'
}
"""

search_msg_2 = "Who is Luning Wang (Tsinghua)"
search_results = search.invoke(search_msg_2)
print(search_results[0])
"""
{
    'url': 'https://wln20.github.io/', 
    'content': "I'm Luning Wang, currently a first-year master student majored in electronic and computer engineering. I'm now actively looking for (research/engineering) intern opportunities in the field of LLMs, MLSys, and potentially other AI & Data-Science related fields! ... [09/2022~06/2024] NICS-EFC, Tsinghua University. I'm open to"
}
"""

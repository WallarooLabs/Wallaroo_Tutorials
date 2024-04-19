from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
from scrapy.selector import Selector
from scrapy.item import Item, Field
from urls import url_list

class LinkItems(Item):
    referer =Field()
    response= Field()
    status = Field()

class LinkSpider(CrawlSpider):
    name = "image-link-web-crawler"
    target_domains = ["localhost:1313"] # list of domains to crawl
    start_urls = url_list # list of starting urls to crawl
    handle_httpstatus_list = [404,410,301,500]

    # Delay requests so sites aren't overwhelmed
    custom_settings = {
        'CONCURRENT_REQUESTS': 2,
        'DOWNLOAD_DELAY': 0.5
    }

    rules = [
        Rule(
            LinkExtractor( allow_domains=target_domains, deny=('patterToBeExcluded'), unique=('Yes')), 
            callback='parse_my_url',
            follow=True),
        # crawl external links and images
        Rule(
            LinkExtractor( allow=(''),deny=("patterToBeExcluded"),deny_extensions=set(), tags = ('img',),attrs=('src',),unique=('Yes')),
            callback='parse_my_url',
            follow=False)
    ]

    def parse_my_url(self, response):
      report_if = [404] 
      if response.status in report_if:
          item = LinkItems()
          item['referer'] = response.request.headers.get('Referer', None)
          item['status'] = response.status
          item['response']= response.url
          yield item
      yield None # if the response did not match return empty
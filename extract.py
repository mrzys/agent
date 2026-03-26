import trafilatura

if __name__ == "__main__":
    downloaded = trafilatura.fetch_url(
        "https://tw.news.yahoo.com/%E5%91%A8%E6%9D%B0%E5%80%AB%E6%96%B0%E5%B0%88%E8%BC%AF%E5%B0%87%E8%88%89%E8%BE%A6%E8%A6%8B%E9%9D%A2%E6%9C%83-%E6%99%82%E9%96%93%E5%9C%B0%E9%BB%9E%E6%9B%9D%E5%85%89%E7%B2%89%E7%B5%B2%E5%BF%AB%E6%90%B6-%E5%8F%AF%E6%90%B6%E5%85%88%E7%9C%8B%E5%A4%A7%E9%8A%80%E5%B9%95%E6%92%ADmv-044200931.html"
    )
    resp = trafilatura.extract(
        downloaded,
        include_formatting=True,
    )
    print(resp)

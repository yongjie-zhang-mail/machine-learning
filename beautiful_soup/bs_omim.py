
import requests
from bs4 import BeautifulSoup
import pandas as pd

class Omim:
    def estrip(self, string):
        if string:
            return string.strip()
        else:
            return None
    def __init__(self):
        self.list_url = 'https://www.ncbi.nlm.nih.gov/omim'
        self.detail_url = 'https://omim.org/entry/'
        self.headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 \
                        (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        self.headers2 = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0'}


    def get_omim_list_by_page(self, page_size, page_num):
        page_rows = []
        data = {
            "term": "autism+",
            "EntrezSystem2.PEntrez.Omim.Omim_ResultsPanel.Entrez_DisplayBar.sPageSize": page_size,
            "EntrezSystem2.PEntrez.Omim.Omim_ResultsPanel.Entrez_DisplayBar.PageSize": page_size,
            "EntrezSystem2.PEntrez.Omim.Omim_ResultsPanel.Entrez_Pager.CurrPage": page_num
        }
        r_list = requests.post(url=self.list_url, headers=self.headers2, data=data)
        html_doc_list = r_list.text
        soup_list = BeautifulSoup(html_doc_list, 'html.parser')
        a_list = soup_list.select('div.rslt > p > a')
        uid_list = [a.get('href').split('/')[-1] for a in a_list]
        # print(len(uid_list))
        # print(uid_list)
        for uid in uid_list:
            rows = self.get_omim_detail(uid)
            if rows:
                page_rows.extend(rows)
        # print(len(page_rows))
        # print(page_rows)
        return page_rows

    def get_omim_detail2(self, uid):
        # https://omim.org/entry/300495
        rows = []
        url = self.detail_url + uid
        r = requests.get(url=url, headers=self.headers)
        html_doc = r.text
        soup = BeautifulSoup(html_doc, 'html.parser')

        trs = soup.select('#mimContent > div > div > div > div > div > table > tbody > tr')

        if trs:
            for tr in trs:
                if tr.select_one('td:nth-child(1) > span > a'):
                    row = {
                        'uid': uid,
                        'title': soup.select_one('#mimFloatingTocMenu > p > span').text.strip(),
                        'location': tr.select_one('td:nth-child(1) > span > a').text.strip(),
                        'phenotype': tr.select_one('td:nth-child(2) > span').text.strip(),
                        'phenotype_mim_number': tr.select_one('td:nth-child(3) > span > a').text.strip(),
                        'inheritance': tr.select_one('td:nth-child(4) > span').text.strip(),
                        'phenotype_mapping_key': tr.select_one('td:nth-child(5) > span > abbr').text.strip(),
                    }
                    print(row)
                    rows.append(row)

        return rows

    def get_omim_total(self):
        total_rows = []
        total_num = 640
        page_size = 200
        page_num = (total_num + page_size - 1) // page_size
        # print(page_num)
        for i in range(1, page_num + 1):
            print(i)
            page_rows = self.get_omim_list_by_page(page_size=page_size, page_num=i)
            total_rows.extend(page_rows)
        # print(len(total_rows))
        # print(total_rows)
        return total_rows



    def get_omim_detail(self, uid):
        # https://omim.org/entry/300495
        rows = []
        url = self.detail_url + uid
        r = requests.get(url=url, headers=self.headers)
        html_doc = r.text
        soup = BeautifulSoup(html_doc, 'html.parser')

        trs = soup.select('#mimContent > div > div > div > div > div > table > tbody > tr')

        if trs:
            for tr in trs:
                if tr.select_one('td:nth-child(1) > span > a'):
                    row = {
                        'uid': uid,
                        'title': soup.select_one('#mimFloatingTocMenu > p > span').text.strip(),
                        'location': tr.select_one('td:nth-child(1) > span > a').text.strip(),
                        'phenotype': tr.select_one('td:nth-child(2) > span').text.strip(),
                        'phenotype_mim_number': tr.select_one('td:nth-child(3) > span > a').text.strip(),
                        'inheritance': tr.select_one('td:nth-child(4) > span').text.strip(),
                        'phenotype_mapping_key': tr.select_one('td:nth-child(5) > span > abbr').text.strip(),
                    }

                    cs = soup.select_one('#mimClinicalSynopsisFold > div')
                    if cs:
                        # mim-font
                        spans = cs.find_all('span', {'class': 'mim-font'})
                        span_texts = [span.get_text().strip() for span in spans]
                        # span_texts = [span.string or next(span.stripped_strings, None) for span in spans]
                        row.__setitem__('clinical_synopsis', span_texts)
                    else:
                        row.__setitem__('clinical_synopsis', '')

                    print(row)
                    rows.append(row)

        return rows
    def save_to_excel(self):
        # total_rows = self.get_omim_list_by_page(page_size=10, page_num=1)
        total_rows = self.get_omim_total()
        df = pd.DataFrame(total_rows)
        df.to_excel('omim.xlsx', index=False)


if __name__ == '__main__':
    omim = Omim()
    #  610908 300495
    # omim.get_omim_detail('608049')
    # omim.get_omim_list_by_page(page_size=20, page_num=1)
    # omim.get_omim_total()
    omim.save_to_excel()




















import readline
from nbformat import read
import requests, xmltodict, json
from xml.dom.minidom import parse, parseString
from bs4 import BeautifulSoup

#headers = {'Accept-Encoding': 'identity'}
"""
GET SMILE
url = lambda x:'https://bitterdb.agri.huji.ac.il/smiFiles/index.php?file=' + str(x) + '_bitterdb.smi'
r = requests.get(url(i), verify = False)
data = r.content
print(r.text)

"""

def GetLigands(id):
  url = 'https://bitterdb.agri.huji.ac.il/Receptor.php?id=' + str(id)
  res = requests.get(url, verify=False)
  data = res.content
  #print(type(data), res.encoding)
  sub = data.decode('iso-8859-1')
  soup = BeautifulSoup(sub, 'html.parser')
  elements = soup.table
  a_list = elements.findChildren('a')
  
  ligands = []
  for a in a_list:
    ligands.append([a['href'], a.contents])
  return(ligands)



#ligands = GetLigands(1)
#print(ligands)

url = 'https://bitterdb.agri.huji.ac.il/dbbitter.php#compoundBrowse'
print(url, '\n\n\n')
res = requests.get(url, verify=False)
data = res.content
#print(type(data), res.encoding)
sub = data.decode('iso-8859-1')
soup = BeautifulSoup(sub, 'html.parser')
elements = soup.find_all('tbody')
print(len(elements))
print(elements, '\n\n\n\n\n')

"""a_list = elements.findChildren('a')

ligands = []
for a in a_list:
  ligands.append([a['href'], a.contents])

print(ligands)
"""
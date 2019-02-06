import os
from os import listdir
from os.path import isfile, join
import json
import sys
# word encoding
import codecs
import time
import operator
# regular expression
import re
# natural langauage toolkit
import nltk

all_files_dir = 'allfiles'

def clean_string(body):
	if len(body) == 0:
		return body
	body = re.sub(r"<!\-\-(.*)?\-\->", "", body)
	while "<script" in body:
		pos1 = body.find("<script ")
		if pos1 == -1:
			pos1 = body.find("<script>")
		if pos1 > -1:
			pos2 = body.find("</script>", pos1 + 3)
			if pos2 > -1:
				body = body[0:pos1] + body[pos2 + 9:]
		#body = re.sub(r"<script(.*)?>(.*)?</script>", "", body)
	body = re.sub(r"<link (.*)?>(.*?)</link>", "", body)
	body = re.sub(r"<link (.*)?>", "", body)

	body = re.sub(r"<a href=([^>])*>", "", body)
	body = re.sub(r"</a>", "", body)
	body = re.sub(r"<\!\-\-widget\-([^>])*>", "", body)
	body = re.sub(r"<\!\-\-callout\-([^>])*>", "", body)
	body = re.sub(r"<\!\-\-image\-([^>])*>", "", body)
	body = re.sub(r"</?([^>]*)>", " ", body)
	body = body.replace("\r\n", " ")
	body = body.replace("\"", "").replace("&nbsp;", "").replace("--", "").replace("''", "")
	body = re.sub(r"[^a-zA-Z\s'\-\.]", " ", body)
	toks = body.split()
	clean_toks = []
	for i in range(len(toks)):
		tok = toks[i]
		len1 = len(tok)
		if tok.find("'") == 0 or tok.find("-") == 0  or tok.find(".") == 0:
		  tok = tok[1:]
		  len1 -= 1
		if tok.find("'") == len1-1 or tok.find("-") == len1-1 or tok.find(".") == len1-1:
		  tok = tok[0:len1-1]
		if len(tok) < 3:
		  continue
		clean_toks.append(tok)

	body = ' '.join(clean_toks)
	return body.lower()

def create_article_files(base_dir, consolidated_file, prefix, start):
	dupnids = {}
	cntr = 0
	for f in listdir(base_dir):
		fullName = base_dir + "/" + f
		if isfile(fullName):
			#outline = "\n\n" + f + "\n-------------------\n"
			outline = "\n\n"
			try:
				tmp = open(fullName, 'r')#codecs.open(fullName, 'r', 'utf-8')
				hlArticle = json.load(tmp)
				tmp.close()

				hlCurArticle = hlArticle

				try:
					nid = "1234567"
					if 'nid' in hlCurArticle:
						nid = str(hlCurArticle['nid'])
					tmpFilename = prefix + "-" + nid + ".json"
					if f != tmpFilename:
						continue
					print(f)
					if nid not in dupnids.keys():
						dupnids[nid] = 1
					else:
						continue
					cntr = cntr + 1
					title = ""
					try:
						if 'title' in hlCurArticle:
							title = hlCurArticle['title']#.encode('utf-8')
						title = clean_string(title)
						print('title: ' + title)
					except Exception as e:
						print(str(e))
					summary = ""
					try:
						if 'serp_summary' in hlCurArticle:
							summary = hlCurArticle['serp_summary']#.encode('utf-8')
						summary = clean_string(summary)
						print('summary: ' + summary)
					except Exception as e:
						print(str(e))
					htmlDescTag = ""
					try:
						if 'htmlDescTag' in hlCurArticle:
							htmlDescTag = hlCurArticle['htmlDescTag']#.encode('utf-8')
						htmlDescTag = clean_string(htmlDescTag)
						print('desc: ' + htmlDescTag)
					except Exception as e:
						print(str(e))
					if len(summary) == 0:
						summary = htmlDescTag
					
					body = ""
					try:
						if 'body' in hlCurArticle:
							body = hlCurArticle['body']#.encode('utf-8')
						body = clean_string(body)
						print('body: ' + body[0:100])
					except Exception as e:
						print(str(e))
					nonanalyzedbody = ""
					try:
						if 'nonanalyzedbody' in hlCurArticle:
							nonanalyzedbody = hlCurArticle['nonanalyzedbody']#.encode('utf-8')
						nonanalyzedbody = clean_string(nonanalyzedbody)
						print('nonanalyzedbody: ' + nonanalyzedbody[0:100])
					except Exception as e:
						print(str(e))
					outline += str(title) + " "
					outline += str(summary) + " "
					outline += str(body) + " "
					outline += str(nonanalyzedbody) + " "
				except:
					dummy = 1
					print('parse error')
			except:
				dummy = 1
				print('json file open error')
				
			try:
				if len(outline) > 10:
					print('writing to consolidated file')
					consolidated_file.write(outline + " ")
			except:
				dummy = 1
			
			try:
				if len(outline) > 10:
					print('writing single file')
					singlefile = open(all_files_dir + '/' + str(f), 'w')
					singlefile.write(outline)
					print(outline[0:100])
					singlefile.close()
					print('success\n-------------------')
			except:
				dummy = 1
		else:
			continue
	return cntr

if __name__ == '__main__':
	cons_file = open('training_data.txt', 'w')
	dirs = ["healthfeature", "authoritynutrition", "partner_article", "newsarticles"]
	start = 0
	for i in range(len(dirs)):
		dir = dirs[i]
		print('---------------------------')
		count = create_article_files("/mnt/data/prod_cms/articles/"+dir, cons_file, dir, start)
		#count = create_article_files(dir, cons_file, dir, start)
		start = start + count
	cons_file.close()

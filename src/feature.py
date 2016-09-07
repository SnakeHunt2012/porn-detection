#!/usr/bin/env python
# -*- coding: utf-8 -*-

from re import compile
from json import dumps, loads
from jieba import cut, add_word
from codecs import open
from pickle import dump
from urlparse import urlparse
from argparse import ArgumentParser
from numpy.linalg import norm
from scipy.sparse import coo_matrix

def build_category_dict(category_file):

    parent_map = {} # sub-tag -> parent-tag
    with open(category_file, 'r') as fd:
        for line in fd:
            line = line.strip()
            if len(line.split('\t')) == 1:
                if line in parent_map: # is sub-line
                    continue
                parent_map[line] = line
            elif len(line.split('\t')) == 2:
                parent_tag, sub_tag = line.split('\t')
                if sub_tag in parent_map:
                    #assert parent_map[sub_tag] == parent_tag
                    pass
                else:
                    parent_map[sub_tag] = parent_tag
            else:
                assert False
                
    parent_list = list(set(parent_map[key] for key in parent_map))
    for tag in parent_list:
        if tag not in parent_map:
            parent_map[tag] = tag
    
    tag_label_dict = dict((parent_list[index], index) for index in xrange(len(parent_list))) # tag -> label
    label_tag_dict = dict((value, key) for key, value in tag_label_dict.iteritems())         # label -> tag
    with open("label-dict.tsv", 'w') as fd:
        fd.write(dumps({
            "parent_map": parent_map,
            "tag_label_dict": tag_label_dict,
            "label_tag_dict": label_tag_dict}, indent=4, ensure_ascii=False
        ))
    return parent_map, tag_label_dict, label_tag_dict

def load_category_dict(label_file):

    with open(label_file, 'r') as fd:
        label_dict = loads(fd.read())
    parent_map = dict((key.encode("utf-8"), value.encode("utf-8")) for key, value in label_dict["parent_map"].iteritems())
    tag_label_dict = dict((key.encode("utf-8"), int(value)) for key, value in label_dict["tag_label_dict"].iteritems())
    label_tag_dict = dict((int(key), value.encode("utf-8")) for key, value in label_dict["label_tag_dict"].iteritems())
    return parent_map, tag_label_dict, label_tag_dict

def load_template_dict(template_file):

    with open(template_file, 'r') as fd:
        template_dict = loads(fd.read())
    word_idf_dict = dict((key.encode("utf-8"), value) for key, value in template_dict["word_idf_dict"].iteritems())
    word_index_dict = dict((key.encode("utf-8"), value) for key, value in template_dict["word_index_dict"].iteritems())
    assert len(word_idf_dict) == len(word_index_dict)
    return word_idf_dict, word_index_dict

def load_netloc_dict(netloc_file):

    netloc_dict = None
    if netloc_file is None:
        return None, None
    with open(netloc_file, 'r') as fd:
        netloc_dict = loads(fd.read())
        netloc_index_dict = netloc_dict["netloc_index_dict"]
        index_netloc_dict = dict((int(key), value) for key, value in netloc_dict["index_netloc_dict"].iteritems())
    return netloc_index_dict, index_netloc_dict

def main():

    parser = ArgumentParser()
    parser.add_argument("label_file", help = "label dict file in json format (input)")
    parser.add_argument("template_file", help = "idf_dict and tempalte_dict in json format (input)")
    parser.add_argument("corpus_file", help = "corpus file (input)")
    parser.add_argument("data_file", help = "data file in pickle format {'url_list': [], 'label_list': [], 'feature_matrix': coo_matrix} (output)")
    parser.add_argument("--netloc-file", help = "netloc file in json format {'netloc_index_dict': {...}, 'index_netloc_dict': {...}}")
    args = parser.parse_args()

    label_file = args.label_file
    template_file = args.template_file
    corpus_file = args.corpus_file
    data_file = args.data_file
    netloc_file = args.netloc_file

    tag_search = compile("(?<=<LBL>).*(?=</LBL>)")
    title_search = compile("(?<=<TITLE>).*(?=</TITLE>)")
    url_search = compile("(?<=<URL>).*(?=</URL>)")
    content_search = compile("(?<=<CONTENT>).*(?=</CONTENT>)")
    image_search = compile("(?<=\[img\])[^\[\]]+(?=\[/img\])")
    image_sub = compile("\[img\][^\[\]]+\[/img\]")
    br_sub = compile("\[br\]")

    parent_map, tag_label_dict, label_tag_dict = load_category_dict(label_file)
    word_idf_dict, word_index_dict = load_template_dict(template_file)
    netloc_index_dict, index_netloc_dict = load_netloc_dict(netloc_file)

    #for word in word_idf_dict:
    #    add_word(word)
    
    url_list = []
    label_list = []
    row_list = []
    column_list = []
    value_list = []
    with open(corpus_file, "r") as fd:
        row_index = 0
        for line in fd:
            result = tag_search.search(line)
            if not result:
                continue
            tag = result.group(0)
    
            result = title_search.search(line)
            if not result:
                continue
            title = result.group(0)
    
            result = url_search.search(line)
            if not result:
                continue
            url = result.group(0)
            
            result = content_search.search(line)
            if not result:
                continue
            content = result.group(0)
            
            content = image_sub.sub("", content)
            content = br_sub.sub("", content)

            label = None
            if len(tag.split('|')) == 1:
                tag = parent_map[tag]
                label = tag_label_dict[tag]
            elif len(tag.split('|')) == 2:
                tag = parent_map[tag.split('|')[0]]
                label = tag_label_dict[parent_map[tag.split('|')[0]]]
            if label == None:
                continue

            content_seg_list = [seg.encode("utf-8") for seg in cut(content)]
            title_seg_list = [seg.encode("utf-8") for seg in cut(title)]

            title_word_tf_dict = {}
            # title -> title_word_tf_dict
            for word in title_seg_list: # word -> term count
                if word not in title_word_tf_dict:
                    title_word_tf_dict[word] = 0
                title_word_tf_dict[word] += 10
            # tc -> tf
            term_count = sum(title_word_tf_dict.itervalues())
            if term_count > 0: # word -> term frequency
                for word in title_word_tf_dict:
                    title_word_tf_dict[word] /= float(term_count)
                    
            content_word_tf_dict = {}
            # content -> content_word_tf_dict
            for word in content_seg_list: # word -> term count
                if word not in content_word_tf_dict:
                    content_word_tf_dict[word] = 0
                content_word_tf_dict[word] += 1
            # tc -> tf
            term_count = sum(content_word_tf_dict.itervalues())
            if term_count > 0: # word -> term frequency
                for word in content_word_tf_dict:
                    content_word_tf_dict[word] /= float(term_count)
                
            title_feature_dict = {}
            for word in title_word_tf_dict:
                if (word in word_idf_dict) and (word in word_index_dict):
                    title_feature_dict[word_index_dict[word]] = title_word_tf_dict[word] * word_idf_dict[word]
            if len(title_feature_dict) > 0:
                feature_norm = norm([value for key, value in title_feature_dict.iteritems()])
                for word in title_feature_dict:
                    title_feature_dict[word] /= feature_norm

            content_feature_dict = {}
            for word in content_word_tf_dict:
                if (word in word_idf_dict) and (word in word_index_dict):
                    content_feature_dict[word_index_dict[word]] = content_word_tf_dict[word] * word_idf_dict[word]
            if len(content_feature_dict) > 0:
                feature_norm = norm([value for key, value in content_feature_dict.iteritems()])
                for word in content_feature_dict:
                    content_feature_dict[word] /= feature_norm

            for column_index in title_feature_dict:
                row_list.append(row_index)
                column_list.append(column_index)
                value_list.append(title_feature_dict[column_index])
            for column_index in content_feature_dict:
                row_list.append(row_index)
                column_list.append(len(word_idf_dict) + column_index)
                value_list.append(content_feature_dict[column_index])
            if len(content_feature_dict) + len(content_feature_dict) == 0:
                row_list.append(row_index)
                column_list.append(0)
                value_list.append(0.0)
            if netloc_file is not None:
                netloc = urlparse(url).netloc
                if netloc in netloc_index_dict:
                    row_list.append(row_index)
                    column_list.append(len(word_idf_dict) + len(word_idf_dict) + netloc_index_dict[netloc])
                    value_list.append(1)
                    
            url_list.append(url)
            label_list.append(label)
            row_index += 1
            # debug
            #print "%s\t%d\t%f\t%f\t%d" % (url, len(word_tf_dict), feature_norm, value_list[-1], label)
        assert len(url_list) == len(label_list) == len(set(row_list))
            
    if netloc_file is not None:
        feature_matrix = coo_matrix((value_list, (row_list, column_list)), shape=(len(url_list), len(word_index_dict) + len(word_index_dict) + len(netloc_index_dict)))
    else:
        feature_matrix = coo_matrix((value_list, (row_list, column_list)), shape=(len(url_list), len(word_index_dict) + len(word_index_dict)))

    with open(data_file, "wb") as fd:
        dump({"url_list": url_list, "label_list": label_list, "feature_matrix": feature_matrix}, fd)

if __name__ == "__main__":

    main()

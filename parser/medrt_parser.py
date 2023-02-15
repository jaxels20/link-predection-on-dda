"""MED-RT parser
   __author__ = "Tenzen Rabgang and Romana Pernisch"
"""

import xmltodict
import os
import sys
from collections import OrderedDict
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import logging
logging.basicConfig(filename='medrt_parser.log', level=logging.DEBUG)


def start_parser(args):
	xml_file_path = args.input
	xml_name = xml_file_path.split('/')[-1].rsplit('.', 1)[0]

	xml_dict = None
	with open(xml_file_path, 'r') as xml:
		xml_dict = xmltodict.parse(xml.read())


	# (1) extract data from XML
	namespace = xml_dict['terminology']['namespace']
	ref_namespaces = xml_dict['terminology']['referencedNamespace']

	proptypes = xml_dict['terminology']['proptype']
	assntypes = xml_dict['terminology']['assntype'] # relation names e.g. may_treat
	qualtypes = xml_dict['terminology']['qualtype']

	terms = xml_dict['terminology']['term'] # name of concepts in MEDRT
	concepts = xml_dict['terminology']['concept'] # detailed description of concept

	associations = xml_dict['terminology']['association'] # detailed description of relations
	logging.debug('XML extracted')


	# (2) extract may_treat and parent-child relations
	may_treat_assoc = []
	parent_of_assoc = []

	for assoc in associations:
		if assoc['name'] == "may_treat":
			may_treat_assoc.append(assoc)
		elif assoc['name'] == "Parent Of":
			parent_of_assoc.append(assoc)


	# (3) add resp. infer may_treat relation from parent to children
	#     for drugs as well as diseases
	parent_child_map = {}

	for pc_assoc in parent_of_assoc:
		parent_child_map[pc_assoc['from_code']] = pc_assoc

	may_treat_assoc_inferred = may_treat_assoc.copy()

	for mt_assoc in may_treat_assoc:
		from_code = mt_assoc['from_code'] # drug
		to_code = mt_assoc['to_code'] # disease

		while from_code in parent_child_map:
			assoc = parent_child_map[from_code]
			
			new_mt_assoc = OrderedDict()
			new_mt_assoc['from_namespace'] = assoc['to_namespace']
			new_mt_assoc['from_code'] = assoc['to_code']
			new_mt_assoc['to_namespace'] = mt_assoc['to_namespace']
			new_mt_assoc['to_code'] = mt_assoc['to_code']
			may_treat_assoc_inferred.append(new_mt_assoc)

			from_code = assoc['to_code']

		while to_code in parent_child_map:
			assoc = parent_child_map[to_code]
			
			new_mt_assoc = OrderedDict()
			new_mt_assoc['to_namespace'] = assoc['to_namespace']
			new_mt_assoc['to_code'] = assoc['to_code']
			
			new_mt_assoc['from_namespace'] = mt_assoc['from_namespace']
			new_mt_assoc['from_code'] = mt_assoc['from_code']
			may_treat_assoc_inferred.append(new_mt_assoc)

			to_code = assoc['to_code']


	# (4) add relations to list/dict and sort by namespace
	code_map = {}
	relations = []

	for mt_assoc in may_treat_assoc_inferred:
		code_map[mt_assoc['from_code']] = { 'code': mt_assoc['from_code'], 'ns': mt_assoc['from_namespace'], 'type': 'drug' }
		code_map[mt_assoc['to_code']] = { 'code': mt_assoc['to_code'], 'ns': mt_assoc['to_namespace'], 'type': 'disease' }

		relations.append((mt_assoc['from_code'], mt_assoc['to_code']))


	drugs = [d for d in code_map.values() if d['type'] == 'drug']
	diseases = [d for d in code_map.values() if d['type'] == 'disease']

	# sort by namespace
	drugs = sorted(drugs, key=lambda k: k['ns'])
	diseases = sorted(diseases, key=lambda k: k['ns']) 


	# (5) write to node list and edgelist
	code_to_id_map = {}
	with open('{}/{}_node_list.txt'.format(args.nodelist_output, xml_name), 'w') as node_list:
		for idx, drug in enumerate(drugs):
			node_list.write('{} {} {} {}\n'.format(idx, drug['code'], drug['type'], drug['ns']))
			code_to_id_map[drug['code']] = idx

		for idx, disease in enumerate(diseases):
			cont_id = idx + len(drugs)
			node_list.write('{} {} {} {}\n'.format(cont_id, disease['code'], disease['type'], disease['ns']))
			code_to_id_map[disease['code']] = cont_id
	logging.debug('Nodelist created')

	with open('{}/{}.edgelist'.format(args.edges_output, xml_name), 'w') as edge_file:
		for rel in relations:
			drug_id = code_to_id_map[rel[0]]
			disease_id = code_to_id_map[rel[1]]

			edge_file.write("{} {}\n".format(drug_id, disease_id))
	logging.debug('Edgelist created')

def main():
	parser = ArgumentParser("MEDRT Parser",
							formatter_class=ArgumentDefaultsHelpFormatter,
							conflict_handler='resolve')

	parser.add_argument('--input', default=r'./xml/Core_MEDRT_2018.03.05_XML.xml', 
						help='Path for XML file to parse.')

	parser.add_argument('--nodelist-output', default=r'./node_lists',
						help="Path for node list.")

	parser.add_argument('--edges-output', default=r'./edges',
						help="Path for edge list.")

	args = parser.parse_args()
	start_parser(args)

if __name__ == "__main__":
	logging.debug('Start process')
	sys.exit(main())
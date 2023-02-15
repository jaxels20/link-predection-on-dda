"""NDF-RT parser
   __author__ = "Tenzen Rabgang and Romana Pernisch"
"""

import xmltodict
import os
from owlready2 import *
import rdflib
from rdflib.namespace import OWL
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import logging
logging.basicConfig(filename='ndfrt_parser.log', level=logging.DEBUG)

def parse_xml(fn):
	xml_dict = None
	with open(fn, 'r') as xml:
		xml_dict = xmltodict.parse(xml.read())
	return xml_dict

def extract_data(xml):
	xml_data = {}
	xml_data['namespace'] = xml['terminology']['namespaceDef'] # owl:Ontology
	xml_data['roles'] = xml['terminology']['roleDef'] # owl:ObjectProperty
	xml_data['associations'] = xml['terminology']['associationDef'] # owl:ObjectProperty
	xml_data['properties'] = xml['terminology']['propertyDef'] # owl:DatatypeProperty / owl:AnnotationProperty???
	xml_data['concepts'] = xml['terminology']['conceptInf'] # owl:Class
	xml_data['kinds'] = xml['terminology']['kindDef'] # merged to ObjectProperty
	xml_data['qualifiers'] = xml['terminology']['qualifierDef'] # ??? / not needed
	return xml_data

def create_owl_class(onto, concept, _data, owl_classes, owl_roles, ref_value):
	with onto:
		value = ""
		for props in concept['properties']['property']:
			if props['name'] == _data['NUI_Code']:
				value = props['value']

		onto_class = types.new_class(value, (Thing,))
		onto_class.label = concept['name']
		onto_class.code = concept[ref_value]

		# creating annotations/tags
		if concept['properties']:
			for props in concept['properties']['property']:
				code = props['name']
				if code in _data['code_to_property']:
					props_name = _data['code_to_property'][code]['name']
					annot_props = types.new_class(props_name, (AnnotationProperty,))
					setattr(onto_class, props_name, props['value'])

		# if concept.get('associations', None):
		# 	if type(concept['associations']['association']) != list:
		# 		concept['associations']['association'] = [concept['associations']['association']]
		# 	for assoc in concept['associations']['association']:
		# 		code = assoc['name']
		# 		if code in code_assoc_map:
		# 			assoc_name = code_property_map[code]['name']
		# 			annot_props = types.new_class(assoc_name, (AnnotationProperty,))
		# 			setattr(onto_class, assoc_name, assoc['value'])

		owl_classes[concept[ref_value]] = onto_class

	return (owl_classes, owl_roles)

def get_help_data(data, ref_value):
	code_kinds_map = {}

	temp_dict = {}
	temp_dict['head_concepts'] = []
	temp_dict['NUI_Code'] = '' # (unique identifier of NDF-RT)
	temp_dict['kind_to_head_code'] = {}
	temp_dict['code_to_property'] = {}
	temp_dict['code_to_roles'] = {}
	temp_dict['code_to_assoc'] = {}

	for concept in data['concepts']:
		if concept['definingConcepts'] is None:
			temp_dict['head_concepts'].append(concept)
	
	for props in data['properties']:
		temp_dict['code_to_property'][props[ref_value]] = props
		if props['name'] == "NUI":
			temp_dict['NUI_Code'] = props[ref_value]

	for kind in data['kinds']:
		code_kinds_map[kind[ref_value]] = kind

	for head in temp_dict['head_concepts']:
		if head['kind'] in code_kinds_map:
			temp_dict['kind_to_head_code'][head['kind']] = head[ref_value]

	for __role in data['roles']:
		temp_dict['code_to_roles'][__role[ref_value]] = __role

	for assoc in data['associations']:
		temp_dict['code_to_assoc'][assoc[ref_value]] = assoc

	return temp_dict

def create_ontology(ont_name, data, is_prev_version):
	logging.debug('--- start ontology creation ---')
	ref_value = "id" if is_prev_version else "code"

	onto = get_ontology(ont_name)
	help_data = get_help_data(data, ref_value)

	# create data properties
	owlProps = {}
	for props in data['properties']:
		with onto:
			code = types.new_class('code', (AnnotationProperty,))

			if is_prev_version:
				props['name'] = props['name'].replace(' ', '_')

			onto_data_property = types.new_class(props['name'],  (DataProperty,))
			onto_data_property.label = props['name']
			onto_data_property.code = props[ref_value]

			owlProps[props['name']] = onto_data_property
	logging.debug('properties created...')

	# create object properties
	owlRoles = {}
	for role in data['roles']:
		with onto:
			onto_data_property = types.new_class(role[ref_value],  (ObjectProperty,))
			onto_data_property.code = role[ref_value]
			onto_data_property.label = role['name']

			owlRoles[role[ref_value]] = onto_data_property
	logging.debug('roles created...')

	for assoc in data['associations']:
		with onto:
			onto_data_property = types.new_class(assoc[ref_value],  (ObjectProperty,))
			onto_data_property.code = assoc[ref_value]
			onto_data_property.label = assoc['name']

			owlRoles[assoc[ref_value]] = onto_data_property
	logging.debug('associations created...')

	# create top-most classes
	owlClasses = {}
	for concept in help_data['head_concepts']:
		owlClasses, owlRoles = create_owl_class(onto, concept, help_data, owlClasses, owlRoles, ref_value)

	for concept in data['concepts']:
		if concept[ref_value] in owlClasses:
			# already processed
			continue
		else:
			owlClasses, owlRoles = create_owl_class(onto, concept, help_data, owlClasses, owlRoles, ref_value)
	logging.debug('classes created...')

	# add additional data to classes
	for concept in data['concepts']:
		current_class = owlClasses[concept[ref_value]]

		if concept['definingConcepts']:
			if type(concept['definingConcepts']['concept']) == str:
				concept['definingConcepts']['concept'] = [concept['definingConcepts']['concept']]

			for conc in concept['definingConcepts']['concept']:
				current_class.is_a.append(owlClasses[conc])

			current_class.is_a.remove(Thing)

		# add relations
		if concept['definingRoles']:
			_role = concept['definingRoles']['role']
			concept['definingRoles']['role'] = [_role] if not type(_role) == list else _role

			for role in concept['definingRoles']['role']:
				if role['name'] in help_data['code_to_roles'] and role['value'] in owlClasses:
					if role['name'] in owlRoles:
						new_prop = owlRoles[role['name']]
						current_class.is_a.append(new_prop.some(owlClasses[role['value']]))
	logging.debug('relations added...')

	# add domain/range to roles/ObjectProperty
	for role in data['roles']:
		with onto:
			role_props = owlRoles[role[ref_value]]
			if role['domain'] in help_data['kind_to_head_code']:
				code_str = help_data['kind_to_head_code'][role['domain']]
				if code_str in owlClasses:
					role_props.domain = owlClasses[code_str]
			if role['range'] in help_data['kind_to_head_code']:
				code_str = help_data['kind_to_head_code'][role['range']]
				if code_str in owlClasses:
					role_props.range = owlClasses[code_str]

	logging.debug('--- end ontology creation ---')

	return onto

def extract_relations_from_owl(owl_fp, onto_n, args):
	g = rdflib.Graph()
	g.parse(owl_fp, format='application/rdf+xml')
	g.bind("owl", OWL)

	owl_path = "{}/{}".format(args.owl_ref, onto_n)
	may_treat_ndfrt_pre = rdflib.URIRef("{}#35".format(owl_path))
	may_treat_ndfrt = rdflib.URIRef("{}#C34".format(owl_path))
	may_treat_fmtsme = rdflib.URIRef("{}#R84".format(owl_path))
	relations = g.query('\
		 SELECT DISTINCT ?drug ?disease \
		 WHERE { \
		 	?rel rdf:type owl:Restriction. \
		 	?drug rdfs:subClassOf ?rel. \
		 	?rel owl:someValuesFrom ?disease. \
		 	?rel owl:onProperty ?code. \
		 	FILTER( ?code IN (?code_ndfrt, ?code_ndfrt_pre, ?code_fmtsme) ) \
		 }', initBindings = {'code_ndfrt': may_treat_ndfrt, 'code_ndfrt_pre': may_treat_ndfrt_pre, 'code_fmtsme': may_treat_fmtsme})
	logging.debug('drug-disease relations extracted')

	return relations

def start_parser(args):
	xml_path = args.input
	xml = parse_xml(xml_path)
	logging.debug('XML parsed')

	# (1) extract data from XML
	xml_data = extract_data(xml)

	# (2) create/save OWL file
	onto_name = xml_path.rsplit('.', 1)[0].rsplit('/', 1)[1]
	ontology = create_ontology(onto_name, xml_data, args.prev_version)
	owl_file_path = os.path.join(args.owl_output, "{}.owl".format(onto_name))
	ontology.save(owl_file_path)

	# (3) create/save node list and edgelist
	relations = extract_relations_from_owl(owl_file_path, onto_name, args)
	content = []
	for relation in relations:
		source = relation[0].split('#')[-1]
		target = relation[1].split('#')[-1]

		content.append((source, target))

	# remove duplicates and sort by drugs
	content = list(set(content))
	content.sort(key=lambda tup: tup[0]) 

	# extract drugs/diseases and clean/sort
	drugs = [tupl[0] for tupl in content]
	diseases = [tupl[1] for tupl in content]
	drugs = list(set(drugs))
	diseases = list(set(diseases))
	drugs.sort()
	diseases.sort()

	# create node list with type and assign ID
	nui_map = {}
	with open('{}/{}_node_list.txt'.format(args.nodelist_output, onto_name), 'w') as node_file:
		for idx, drug in enumerate(drugs):
			nui_map[drug] = { 'id': idx, 'type': 'drug' }
			node_file.write('{}\t{}\t{}\n'.format(idx, drug, 'drug'))

		for idx, disease in enumerate(diseases):
			cont_id = idx + len(drugs)
			nui_map[disease] = { 'id': cont_id, 'type': 'disease' }
			node_file.write('{}\t{}\t{}\n'.format(cont_id, disease, 'disease'))
	logging.debug('Nodelist created')


	# replace NUI with ID (from above) for edgelist
	with open('{}/{}.edgelist'.format(args.edges_output, onto_name), 'w') as edge_file:
		for pair in content:
			drug = nui_map[pair[0]]
			disease = nui_map[pair[1]]

			edge_file.write('{} {}\n'.format(drug['id'], disease['id']))
	logging.debug('Edgelist created')

	logging.debug('End process')

def main():
	parser = ArgumentParser("NDFRT Parser",
							formatter_class=ArgumentDefaultsHelpFormatter,
							conflict_handler='resolve')

	parser.add_argument('--input', default=r'./xml/NDFRT_Public_2013.01.07_TDE_inferred.xml', 
						help='Path for XML file to parse (only inferred versions).')
						# if not inferred version, change "conceptInf" to "conceptDef" in line 23

	parser.add_argument('--owl-output', default=r'./owl',
						help="Path for owl file.")

	parser.add_argument('--owl-ref', default='file:///C:/Users/teny_/Documents/2020-tenzen-rabgang/parser/owl',
						help='Absolute path where the owl file is resided.')

	parser.add_argument('--nodelist-output', default=r'./node_lists',
						help="Path for node list.")

	parser.add_argument('--edges-output', default=r'./edges',
						help="Path for edge list.")

	parser.add_argument('--prev-version', default=False, type=bool,
						help="Set this to true for 2009.07.14 and older versions. In older versions ID was taken as ref instead of code...")

	args = parser.parse_args()
	start_parser(args)

if __name__ == "__main__":
	logging.debug('Start process')
	sys.exit(main())

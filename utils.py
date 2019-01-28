import os
import sys
import subprocess

def generate_new_project(directory):
	os.system("mkdir " + new_directory)
	os.system("mkdir " + new_directory + '/transitional')
	os.system("mkdir " + new_directory + '/model')
	os.system("mkdir " + new_directory + '/img')
	os.system('echo " " > ' + new_directory + '/model/history.txt')


if (len(sys.argv) == 1):
	print('no arugments provided. Please run using:')
	print('utils.py enter_name_of_new_directory_here')
else:
	# Generate New Project (change data path as needed)
	new_directory_name = str(sys.argv[1])
	data_path = "data/"
	new_directory_name = str(sys.argv[1])
	new_directory = data_path + new_directory_name
	generate_new_project(new_directory)

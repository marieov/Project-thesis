import config

def read_record_files(): 
    '''
    Open and store the names of all files and of the ones with seizures

        param

        return files: list of names of all the files, on the form 'chb01/chb01_01'
        return files_seizures: list of names of all the files with seizures, on the form 'chb01/chb01_01'
    '''
    records = open(config.DATASET_PATH + '/' + 'RECORDS_TXT.txt', 'r')
    records_seizures = open(config.DATASET_PATH + '/' + 'RECORDS-WITH-SEIZURES_TXT.txt', 'r')
    files = records.read().splitlines()
    files_seizures = records_seizures.read().splitlines()
    records.close()
    records_seizures.close()
    return files, files_seizures
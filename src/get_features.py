from pathlib import Path
from argparse import ArgumentParser

from rbclib import RBCPath
from pathlib import Path
import pandas as pd
import numpy as np

def get_participant_demographics():
    "Get the subject_id, demographic, and behavioral data of train and test participants."

    rbcdata_path = Path('/home/jovyan/shared/data/RBC')
    train_filepath = rbcdata_path / 'train_participants.tsv'
    test_filepath = rbcdata_path / 'test_participants.tsv'
    
    with train_filepath.open('r') as f:
        train_data = pd.read_csv(f, sep='\t')
        train_data['is_train'] = True
        
    with test_filepath.open('r') as f:
        test_data = pd.read_csv(f, sep='\t')
        test_data['is_train'] = False
    
    participant_demographics = pd.concat([train_data, test_data])

    return participant_demographics

def education2num(df, drop=True):
    """Add education as a numerical variable."""

    # ordered 
    education_map = {
        'Complete primary': 1,
        'Complete secondary': 2,
        'Complete tertiary': 3,
        'No/incomplete primary': 0} # 0 is a better option than np.nan, because no education is in itself a category, not missing data
    
    df['parent_1_education_num'] = df['parent_1_education'].map(education_map)
    df['parent_2_education_num'] = df['parent_2_education'].map(education_map)

    if drop:
        df = df.drop(['parent_1_education', 'parent_2_education'], axis=1)

    return df


def load_fsdata(participant_id, local_cache_dir=(Path.home() / 'cache')):
    "Loads and returns the dataframe of a PNC participant's FreeSurfer data."

    # Check that the local_cache_dir exists and make it if it doesn't.
    if local_cache_dir is not None:
        local_cache_dir = Path(local_cache_dir)
        local_cache_dir.mkdir(exist_ok=True)
    
    # Make the RBCPath and find the appropriate file:
    pnc_freesurfer_path = RBCPath(
        'rbc://PNC_FreeSurfer/freesurfer',
        # We provide the local_cache_dir to the RBCPath object; all paths made
        # from this object will use the same cache directory.
        local_cache_dir=local_cache_dir)
    participant_path = pnc_freesurfer_path / f'sub-{participant_id}'
    tsv_path = participant_path / f'sub-{participant_id}_regionsurfacestats.tsv'

    # Use pandas to read in the TSV file:
    with tsv_path.open('r') as f:
        data = pd.read_csv(f, sep='\t')

    # Return the loaded data:
    return data

def get_fs_features_subject(subject_df, atlas='aparc', measure='GrayVol', average=True):
    """
    Get the freesurfer features of a subject. Returns a single row
    dataframe where columns are the region names from the atlas, and the
    values are the measure selected. It will average the value of both regions.
    """

    assert atlas in list(subject_df['atlas'].unique()), "Not a valid atlas :("
    assert measure in subject_df.columns.to_list(), "Not a valid measure :("
    
    # filter by atlas, get one row of mean volume by region
    df_ = subject_df.query(f"atlas=='{atlas}'")
          
    if not average:
        # create a different StructName value per hemisphere, in this case it won't matter if the next code is trying to average
        df_.loc[:,'StructName'] = df_['StructName'] + '_' + df_['hemisphere']
        
    df_ = (df_  
         .groupby(['StructName'])
         [measure]
         .mean()
         .to_frame()
         .transpose()
        )
        
    # some pandas weird things to make the output nice
    df_.columns.name = 'index'
    df_ = df_.reset_index(drop=True)
    # add the subject id to be able to concat data later on
    df_['subject_id'] = subject_df['subject_id'].unique()
    
    return df_

# we put the body of our code here because python nerds say this is best practice
def main():
    parser = ArgumentParser()
    parser.add_argument('output', 
                        help="Path of the output csv file with the subset of features.")
    parser.add_argument('-a', '--atlas', 
                        choices=['aparc', 'Yeo2011_17Networks_N1000', 'glasser', 'AAL', 'PALS_B12_Brodmann'], default='aparc', 
                        help="Name of the freesurfer atlas to get regions from.")
    parser.add_argument('-m', '--measure', 
                        choices=['GrayVol', 'ThickAvg', 'SurfArea'], default='GrayVol', 
                        help="Name of the freesurfer measure to get from the atlas's regions.")
    parser.add_argument('--no-average', dest='average',
                        action='store_false',
                        help="Don't average freesufer measures of both hemispheres.")
    
    parser.set_defaults(average=True)
    
    args = parser.parse_args()

    # assert there is a directory you can write to
    assert Path(args.output).parent.exists(), "Output target directory doesn't exist :O"

    # get non-freesurfer participant data
    participant_demographics = get_participant_demographics()

    # get a list of participant ids
    participants = list(participant_demographics['participant_id'].unique())

    # empty list to store dataframe of single-participant features
    feature_df_list = []

    # loop across subjects
    for participant in participants:
        # load freesurfer data of that subject
        # some participants don't have fs data
        try:
            # if fs file exists, load it
            df_ = load_fsdata(participant)
            print(f"Succesfully loaded sub-{participant}'s data")
        except:
            # if not, skip to next participant instead of raising an error
            print(f"sub-{participant}'s data does not exist. Skipping to next participant.")
            continue
            
        # get freesurfer features of that subject and store them in the list
        feature_df_list.append(get_fs_features_subject(df_, atlas=args.atlas, measure=args.measure, average=args.average))

    # concatenate the single-subject dataframes of freesurfer features 
    feature_df = pd.concat(feature_df_list)

    # add the 'sub-' prefix to the participant id
    participant_demographics['subject_id'] = 'sub-' + participant_demographics['participant_id'].astype('string')
    participant_demographics = participant_demographics.drop('participant_id', axis=1)

    # education to numeric
    participant_demographics = education2num(participant_demographics)
    
    # merge fs features with demographic data
    feature_df = feature_df.merge(participant_demographics, on='subject_id')

    # write data
    feature_df.to_csv(args.output, index=False)

    return feature_df

# also doing this because of python nerds
if __name__ == '__main__':
    main()




    
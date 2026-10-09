"""TrustMR experiments: VSQ vector-query sensitivity to Hamming-radius expansion."""
from run import *
def main():
 assert json.loads((OUT/'trustmr_complete.json').read_text())['complete']
 save('range_sensitivity_amendment.json',{'original_trigger_fired':False,'status':'post-screening exploratory sensitivity, disclosed after exact-match holdout availability','basis':'screening text Recall low at every predefined point','fixed_configurations':{'V3_RANGE':[16,16,1],'V4_RANGE':[32,8,1]},'frozen_exact_recommendation_changed':False,'holdout_run':False,'claim':'not preregistered or confirmatory; all exact-match main results retained'})
 data=base_data()
 for name in ('V3','V4'):
  if (OUT/f'{name}_RANGE_screening_complete.json').exists():continue
  ix=load_index(name,data);execute(ix,name+'_RANGE','screening',radius=1);del ix;gc.collect()
if __name__=='__main__':main()

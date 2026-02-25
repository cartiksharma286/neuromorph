#include <iostream>
#include <math.h>

class pcy {

 public:
  void computeInteraction(vector<float> corticalActivity);

  void computeChange(bool diseased, vector<float> corticalActivity, string clinicalCondition);

  void getDisorder();
  void setDistorder()
    
 private:
    vector<float> m_corticalActivity;
    bool isAnonymized;
 protected:
    

};

#include <iostream.h>


class neuroConditions {

 public:
  void getDisorder();
  int setDisorder();

  bool diagnosis(string disorder);
  bool rehabilitation();
  
 private:
  vector<string> neurologicalDisorders;

  enum var = {'schizophrenia', 'depression', 'dementia', 'stroke', 'autism'};
};

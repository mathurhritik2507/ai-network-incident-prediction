from netops_ai.analytics import generate_data,fit_evaluate

def test_deterministic():
    assert generate_data(1,48).equals(generate_data(1,48))

def test_pipeline():
    model,test,metrics=fit_evaluate(generate_data(3,100))
    assert len(test)>0
    assert 0<=metrics["f1"]<=1

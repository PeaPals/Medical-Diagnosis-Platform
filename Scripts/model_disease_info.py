# Testing Feature :-

#import numpy as np
#from random import randint

#def model_disease_info(model, disease, expander_box):
#    expander_box.subheader(f"This selected task {disease} is using model {model}")
#
#    metrics_viewer = expander_box.columns(10)

#    for i in range(10):
#        metrics_viewer[i].metric("Model Accuracy", f"{randint(70, 100)}%", f"{randint(-15, 15)}%")

#    chart_data = np.random.randn(20, 3)
#    expander_box.area_chart(chart_data)


def model_disease_info(model, model_info, disease, expander_box):
    expander_box.subheader(f"This selected task {disease} is using {model}")
    expander_box.warning(model_info)

    # Disclaimer
    expander_box.info("""Models used under this project are state of the art models available on
            github. We in no way claim the right of development of these models
            in any way. We, in this project are only using these models for predictions.""")
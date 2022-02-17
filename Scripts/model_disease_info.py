import numpy as np
from random import randint

def model_disease_info(model, disease, expander_box):
    expander_box.subheader(f"This selected task {disease} is using model {model}")

    metrics_viewer = expander_box.columns(10)

    for i in range(10):
        metrics_viewer[i].metric("Model Accuracy", f"{randint(70, 100)}%", f"{randint(-15, 15)}%")

    chart_data = np.random.randn(20, 3)
    expander_box.area_chart(chart_data)
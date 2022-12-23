import plotly.graph_objects as go

def show_charts(container, graph_data):

    fig = go.Figure()

    for model_type in graph_data.keys():
        predictions, diseases = graph_data[model_type]
        
        fig.add_trace(go.Bar(
            x=[x[0] for x in diseases],
            y=predictions,
            name=model_type,
        ))

    # Here we modify the tickangle of the xaxis, resulting in rotated labels.
    # fig.update_layout(barmode='group', xaxis_tickangle=-45)

    fig.update_layout(
        title='Disease Predictions',

        xaxis=dict(
            title='Diseases',
            titlefont_size=25,
            tickfont_size=15,
            tickangle=-75

        ),

        yaxis=dict(
            title='Confidence',
            titlefont_size=25,
            tickfont_size=15,
            range=[0, 1]
        ),

        legend=dict(
            # x=0,
            # y=1.0,
            bgcolor='rgba(0, 0, 0, 0)',
            bordercolor='rgba(255, 255, 255, 0)'
        ),

        barmode='group',
        bargap=0.15, # gap between bars of adjacent location coordinates.
        bargroupgap=0.1 # gap between bars of the same location coordinate.
    )

    # fig.update_traces(texttemplate='%{text:.2s}', textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)

    # container.plotly_chart(fig, theme="streamlit", use_container_width=True)
    container.plotly_chart(fig, use_container_width=True)



def show_acc_charts(container, graph_data):

    fig = go.Figure()

    for model_type in graph_data.keys():
        predictions, diseases = graph_data[model_type]
        
        fig.add_trace(go.Bar(
            x=[x[0] for x in diseases],
            y=[x[1] for x in diseases],
            name=f"{model_type} Accuracy",
        ))

    # Here we modify the tickangle of the xaxis, resulting in rotated labels.
    # fig.update_layout(barmode='group', xaxis_tickangle=-45)

    fig.update_layout(
        title='Accuracy Chart',

        xaxis=dict(
            title='Diseases',
            titlefont_size=25,
            tickfont_size=15,
            tickangle=-75

        ),

        yaxis=dict(
            title='Accuracy',
            titlefont_size=25,
            tickfont_size=15,
            range=[0, 1]
        ),

        legend=dict(
            # x=0,
            # y=1.0,
            bgcolor='rgba(0, 0, 0, 0)',
            bordercolor='rgba(255, 255, 255, 0)'
        ),

        barmode='group',
        bargap=0.15, # gap between bars of adjacent location coordinates.
        bargroupgap=0.1 # gap between bars of the same location coordinate.
    )

    # fig.update_traces(texttemplate='%{text:.2s}', textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)

    # container.plotly_chart(fig, theme="streamlit", use_container_width=True)
    container.plotly_chart(fig, use_container_width=True)
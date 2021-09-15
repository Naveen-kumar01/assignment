import fbprophet
from fbprophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv(r'C:\Users\navee\Desktop\final_data.csv')
df.drop(columns=['Unnamed: 0','max_temp','min_temp'])
df['Time']=pd.to_datetime(df['Time'])
df['Total']=df.eval('AC1+AC2+AC3+AC4+AC5+AC6+AC7+AC8+AC9+AC10+AC11+AC12+AC13+AC14+AC15+AC16+AC17+AC18')
df1=df[['Time','Total']]
df2=df1.rename(columns={'Time':'ds','Total':'y'})
df2['y']=df2['y']-df2['y'].shift(1)
model=Prophet()
model.fit(df2)
future_dates=model.make_future_dataframe(periods=30)
pred=model.predict(future_dates)
print(pred)
plt.plot(pred)
model.plot(pred)
pred.to_csv('predicted.csv')
from fbprophet.diagnostics import cross_validation
df_cv=cross_validation(model,horizon="30 days",period='10 days',initial='20 days')

from fbprophet.diagnostics import performance_metrics
df_performance=performance_metrics(df_cv)
df_performance.head()
from fbprophet.plot import plot_cross_validation_metric
fig=plot_cross_validation_metric(df_cv,metric='rmse')
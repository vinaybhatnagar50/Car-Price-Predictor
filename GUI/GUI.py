#  GUI of this Project
from tkinter import *

def show_entry_fields():
  p1 = float(e1.get())
  p2 = float(e2.get())
  p3 = float(e3.get())
  p4 = float(e4.get())
  p5 = float(e5.get())
  p6 = float(e6.get())
  p7 = float(e7.get())

  model = joblib.load('car_dekho_price_predictor')
  data_new = pd.DataFrame({
   'year':p1,
   'km_driven':p2,
   'fuel':p3,
   'seller_type':p4,
   'transmission':p5,
   'owner':p6,
   'Age' :p7,
},index=[0])
  result=model.predict(data_new)
  Label(master, text="Car Purchase Amount").grid(row=9)
  Label(master, text=result).grid(row=10)
  print("Car Purchase Amount",result[0])

master = Tk()
master.title("Car Price Prediction Using Machine Learning")
label = Label(master, text = "Car Price Prediction Using Machine Learning" , 
               bg = "black" , fg = "white").grid(row=0,columnspan=2)

Label(master, text='year').grid(row=1)
Label(master, text='km_driven').grid(row=2)
Label(master, text='fuel').grid(row=3)
Label(master, text='seller_type').grid(row=4)
Label(master, text='transmission').grid(row=5)
Label(master, text='owner').grid(row=6)
Label(master, text='Age').grid(row=7)

e1 = Entry(master)
e2 = Entry(master)
e3 = Entry(master)
e4 = Entry(master)
e5 = Entry(master)
e6 = Entry(master)
e7 = Entry(master)

e1.grid(row=1, column=1)
e2.grid(row=2, column=1)
e3.grid(row=3, column=1)
e4.grid(row=4, column=1)
e5.grid(row=5, column=1)
e6.grid(row=6, column=1)
e7.grid(row=7, column=1)

Button(master, text='Predict', command=show_entry_fields).grid()

mainloop()
import pickle
import pandas as pd
from flask import Flask, render_template, request, send_file


app = Flask(__name__)

@app.route("/")
def home() :
    return render_template("index.html")


@app.route("/about")
def about() :
    return render_template("about.html")

@app.route("/material")
def material() :
    return render_template("material.html")



@app.route("/predict", methods=["GET", "POST"])
def predict() :
    if request.method == 'POST':
        # mess = [float(x) for x in request.form.values()]
        mess = request.form
        val = []
        for key, value in mess.items():
            val.append(float(value))
            
        val = [val]
        model = pickle.load(open("models/logicreg_model.sav", 'rb')) 
        columns = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                    'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
                    'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']
        mess_df = pd.DataFrame(val, columns=columns)
        print(mess_df)
        result = model.predict(mess_df)[0]
        ans = "UNKNOWN"
        if result == 1 :
            ans = "FRAUD"
        else :
            ans = "LEGITIMATE"
        return render_template("result.html", result=ans)
        
    
    
    
    





if __name__ == "__main__" :
    app.run(debug=True)
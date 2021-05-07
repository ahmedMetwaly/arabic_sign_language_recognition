


app = Flask(__name__)



@app.route('/')
@app.home('/home')
def home():
  return "hello"

if __name__ == "__main__":
    app.secret_key = 'ItIsASecret'
    app.debug = True
    app.run()

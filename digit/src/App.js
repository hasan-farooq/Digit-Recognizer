import axios from 'axios';
import './App.css';
import DrawingBoard from "react-drawing-board";

// const DrawingBoard = require('react-drawing-board');

function App() {

  function post(image_url) {
    axios.post('http://localhost:5000/', {url : image_url}).then((res) => {
      console.log(res.data)
      alert("Model Predicted : " + res.data);
    });
  }
  return (
    <div>

      <DrawingBoard
        style={{
          width: "1000px", height: "500px", border: "solid 2px",
          borderRadius: "5px", margin: "10px", marginLeft: "150px"
        }}
        onSave={(image) => post(image.dataUrl)}
      > </DrawingBoard>
    </div>
  );
}

export default App;

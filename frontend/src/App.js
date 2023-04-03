import React, { useState } from "react";
import axios from "axios";
import "./App.css";

export default function Home() {
  const [fileData, setFileData] = useState("");
  const [predict, setPredict] = useState("");

  const getFile = (e) => {
    setFileData(e.target.files[0]);
  };

  const uploadFile = async (e) => {

    e.preventDefault();

    const data = new FormData();
    data.append("file", fileData);

    await axios({
      method: "POST",
      url: "http://localhost:8000/predict",
      data: data,
    }).then((res) => {
      setPredict(JSON.stringify(res.data));
    });
  };

  return (
    <>
      <form onSubmit={uploadFile}>
        <input type="file" name="file" onChange={getFile} required />
        <button type="submit">Upload </button>
      </form>
      {predict.length > 0 && <h2>Prediction = {predict.substring(2,4)}</h2>}
    </>
  );
}

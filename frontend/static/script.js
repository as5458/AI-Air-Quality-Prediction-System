let aqiChart;

function getAQI(){

    const city =
    document.getElementById("cityInput").value;

    if(city===""){
        alert("Enter city");
        return;
    }

    fetch("/predict",{
        method:"POST",

        headers:{
            "Content-Type":"application/json"
        },

        body:JSON.stringify({
            city:city
        })
    })

    .then(response=>response.json())

    .then(data=>{

        if(data.error){
            alert(data.error);
            return;
        }

        document.getElementById(
            "currentAQI"
        ).innerText=data.current_aqi;

        document.getElementById(
            "aqiCircle"
        ).innerText=data.current_aqi;

        document.getElementById(
            "aqiCategory"
        ).innerText=data.category;


        drawChart(
            data.dates,
            data.forecast
        );


        const tbody =
        document.getElementById(
            "pollutantBody"
        );

        tbody.innerHTML="";


        for(let i=0;
            i<data.forecast.length;
            i++){

            const row =
            document.createElement("tr");

            row.innerHTML=`
                <td>Day ${i+1}</td>
                <td>${data.future_components.pm2_5[i].toFixed(2)}</td>
                <td>${data.future_components.pm10[i].toFixed(2)}</td>
                <td>${data.future_components.no2[i].toFixed(2)}</td>
                <td>${data.future_components.co[i].toFixed(2)}</td>
                <td>${data.future_components.o3[i].toFixed(2)}</td>
            `;

            tbody.appendChild(row);
        }

    })

    .catch(error=>{
        console.log(error);
        alert("Prediction failed");
    });

}


function drawChart(labels,values){

    const ctx =
    document.getElementById(
        "aqiChart"
    ).getContext("2d");

    if(aqiChart){
        aqiChart.destroy();
    }

    aqiChart = new Chart(ctx,{

        type:"line",

        data:{
            labels:labels,

            datasets:[{
                label:"Predicted AQI",
                data:values,
                borderColor:"#16a085",
                backgroundColor:"rgba(22,160,133,0.15)",
                borderWidth:3,
                tension:0.4,
                fill:true
            }]
        }
    });
}
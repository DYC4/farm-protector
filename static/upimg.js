document.querySelector("#sendButton").addEventListener('click', () => {
  let selectFile = document.querySelector("#myfile").files[0];

  console.log(selectFile);
})


document.querySelector("#sendButton").addEventListener('click', () => {

  let selectFile = document.querySelector("#myfile").files[0];

  const file = URL.createObjectURL(selectFile);

  document.querySelector(".uploadImage").src = file;

})
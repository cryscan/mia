#[cfg(not(feature = "web"))]
#[inline]
pub fn spawn<O, F>(future: F) -> tokio::task::JoinHandle<O>
where
    O: Send + 'static,
    F: std::future::Future<Output = O> + Send + 'static,
{
    tokio::spawn(future)
}

#[cfg(feature = "web")]
#[inline]
pub fn spawn<F>(future: F)
where
    F: std::future::Future<Output = ()> + 'static,
{
    wasm_bindgen_futures::spawn_local(future);
}

#[cfg(not(feature = "web"))]
pub type BoxFuture<'a, T> = futures::future::BoxFuture<'a, T>;
#[cfg(feature = "web")]
pub type BoxFuture<'a, T> = futures::future::LocalBoxFuture<'a, T>;
